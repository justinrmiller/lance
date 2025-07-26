# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright The Lance Authors

from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Dict,
    Generator,
    Iterable,
    List,
    Literal,
    Optional,
    Tuple,
    Union,
)

import numpy as np

import pyarrow as pa

import lance
from lance.fragment import DEFAULT_MAX_BYTES_PER_FILE, FragmentMetadata, write_fragments

from ..dependencies import ray

if TYPE_CHECKING:
    import pandas as pd

__all__ = ["LanceDatasink", "LanceFragmentWriter", "LanceCommitter", "write_lance"]

NONE_ARROW_STR = "None"


def _pd_to_arrow(
    df: Union[pa.Table, "pd.DataFrame", Dict], schema: Optional[pa.Schema]
) -> pa.Table:
    """Convert a pandas DataFrame to pyarrow Table."""
    from ..dependencies import _PANDAS_AVAILABLE
    from ..dependencies import pandas as pd
    import numpy as np

    if isinstance(df, dict):
        # Handle empty dict case
        if not df:
            if schema:
                # Create empty table with the provided schema
                return pa.Table.from_pydict({field.name: [] for field in schema}, schema=schema)
            else:
                # Return empty table with empty schema
                return pa.Table.from_pydict({})
        
        # Check if dict has all empty arrays/lists
        all_empty = all(
            (isinstance(v, (list, np.ndarray)) and len(v) == 0) or v is None 
            for v in df.values()
        )
        
        if all_empty and schema:
            # Create properly structured empty table
            return pa.Table.from_pydict({field.name: [] for field in schema}, schema=schema)
        
        # Handle the NaN issue for Ray >= 2.38
        cleaned_dict = {}
        for key, value in df.items():
            if isinstance(value, np.ndarray):
                if value.dtype == np.float32 or value.dtype == np.float64:
                    # Check if this should be a string column based on schema
                    if schema and key in [field.name for field in schema if pa.types.is_string(field.type)]:
                        # Convert float NaN back to None for string columns
                        cleaned_value = [None if pd.isna(v) else str(v) for v in value]
                        cleaned_dict[key] = cleaned_value
                    else:
                        cleaned_dict[key] = value
                else:
                    cleaned_dict[key] = value
            elif isinstance(value, (list, np.ndarray)):
                # Handle lists that might contain NaN
                if schema and key in [field.name for field in schema if pa.types.is_string(field.type)]:
                    cleaned_value = [None if pd.isna(v) else v for v in value]
                    cleaned_dict[key] = cleaned_value
                else:
                    cleaned_dict[key] = value
            else:
                cleaned_dict[key] = value
        return pa.Table.from_pydict(cleaned_dict, schema=schema)
    
    elif _PANDAS_AVAILABLE and isinstance(df, pd.DataFrame):
        # Handle empty DataFrame case
        if len(df) == 0:
            if schema:
                # Create empty table with the provided schema
                return pa.Table.from_pydict({field.name: [] for field in schema}, schema=schema)
            else:
                # Let pandas create the schema, then create empty table
                if len(df.columns) > 0:
                    # DataFrame has columns but no rows
                    empty_schema = pa.Schema.from_pandas(df).remove_metadata()
                    return pa.Table.from_pydict({field.name: [] for field in empty_schema}, schema=empty_schema)
                else:
                    # Completely empty DataFrame
                    return pa.Table.from_pydict({})
        
        # Handle NaN in DataFrame for potential string columns
        df_copy = df.copy()
        if schema:
            string_columns = [field.name for field in schema if pa.types.is_string(field.type)]
            for col in string_columns:
                if col in df_copy.columns:
                    # Replace NaN with None for string columns
                    df_copy[col] = df_copy[col].replace({np.nan: None, pd.NaType(): None})
        
        tbl = pa.Table.from_pandas(df_copy, schema=schema)
        if tbl.schema.metadata:
            tbl = tbl.replace_schema_metadata(None)
        return tbl
    
    elif isinstance(df, pa.Table):
        # Handle empty table case
        if len(df) == 0:
            if schema:
                # Don't try to cast empty tables, create new one with correct schema
                return pa.Table.from_pydict({field.name: [] for field in schema}, schema=schema)
            else:
                # Return as-is if no schema provided
                return df
        
        # For non-empty tables, attempt casting if schema provided
        if schema is not None:
            try:
                return df.cast(schema)
            except pa.ArrowInvalid as e:
                # If cast fails due to schema mismatch on empty table
                if len(df) == 0:
                    return pa.Table.from_pydict({field.name: [] for field in schema}, schema=schema)
                # Re-raise for actual schema mismatches on non-empty data
                raise
        return df
    
    return df


def _write_fragment(
    stream: Iterable[Union[pa.Table, "pd.DataFrame"]],
    uri: str,
    *,
    schema: Optional[pa.Schema] = None,
    max_rows_per_file: int = 1024 * 1024,
    max_bytes_per_file: Optional[int] = None,
    max_rows_per_group: int = 1024,
    data_storage_version: Optional[str] = None,
    storage_options: Optional[Dict[str, Any]] = None,
) -> List[Tuple[FragmentMetadata, pa.Schema]]:
    from ..dependencies import _PANDAS_AVAILABLE
    from ..dependencies import pandas as pd

    # Collect all blocks to check if we have any data
    blocks = list(stream)
    if not blocks:
        return []
    
    # Check if all blocks are empty
    all_empty = True
    non_empty_block = None
    
    for block in blocks:
        if _PANDAS_AVAILABLE and isinstance(block, pd.DataFrame):
            if len(block) > 0:
                all_empty = False
                non_empty_block = block
                break
        elif isinstance(block, pa.Table):
            if len(block) > 0:
                all_empty = False
                non_empty_block = block
                break
        elif isinstance(block, dict):
            # Check if dict has any non-empty values
            for v in block.values():
                if hasattr(v, '__len__') and len(v) > 0:
                    all_empty = False
                    non_empty_block = block
                    break
            if not all_empty:
                break
    
    # Now handle schema inference
    if schema is None:
        if all_empty:
            # No data to infer schema from
            return []
        
        # Use the non-empty block to infer schema
        if _PANDAS_AVAILABLE and isinstance(non_empty_block, pd.DataFrame):
            schema = pa.Schema.from_pandas(non_empty_block).remove_metadata()
        elif isinstance(non_empty_block, dict):
            tbl = pa.Table.from_pydict(non_empty_block)
            schema = tbl.schema.remove_metadata()
        elif isinstance(non_empty_block, pa.Table):
            schema = non_empty_block.schema
        else:
            # Try first block as fallback
            first = blocks[0]
            try:
                tbl = _pd_to_arrow(first, None)
                schema = tbl.schema
            except Exception:
                raise ValueError(f"Cannot infer schema from type {type(first)}")
        
        # Safety check for empty schema
        if len(schema) == 0:
            return []
    
    # If all blocks are empty but we have a schema, return empty
    if all_empty:
        return []

    def record_batch_converter():
        for block in blocks:
            try:
                tbl = _pd_to_arrow(block, schema)
                if len(tbl) > 0:  # Only yield non-empty batches
                    yield from tbl.to_batches()
            except Exception as e:
                # Log but skip problematic blocks
                import warnings
                warnings.warn(f"Skipping block due to conversion error: {e}")
                continue

    max_bytes_per_file = (
        DEFAULT_MAX_BYTES_PER_FILE if max_bytes_per_file is None else max_bytes_per_file
    )

    # # Collect batches to check if we have any
    batches = list(record_batch_converter())
    if not batches:
        return []

    reader = pa.RecordBatchReader.from_batches(schema, batches)
    fragments = write_fragments(
        reader,
        uri,
        schema=schema,
        max_rows_per_file=max_rows_per_file,
        max_rows_per_group=max_rows_per_group,
        max_bytes_per_file=max_bytes_per_file,
        data_storage_version=data_storage_version,
        storage_options=storage_options,
    )
    return [(fragment, schema) for fragment in fragments]

class _BaseLanceDatasink(ray.data.datasource.datasink.Datasink):
    """Base Lance Ray Datasink."""

    def __init__(
        self,
        uri: str,
        schema: Optional[pa.Schema] = None,
        mode: Literal["create", "append", "overwrite"] = "create",
        storage_options: Optional[Dict[str, Any]] = None,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        self.uri = uri
        self.schema = schema
        self.mode = mode

        self.read_version: int | None = None
        self.storage_options = storage_options

    @property
    def supports_distributed_writes(self) -> bool:
        return True

    def on_write_start(self):
        if self.mode == "append":
            try:
                ds = lance.LanceDataset(self.uri, storage_options=self.storage_options)
                self.read_version = ds.version
                if self.schema is None:
                    self.schema = ds.schema
            except ValueError:
                # Dataset doesn't exist yet, switch to create mode
                self.mode = "create"

    def on_write_complete(self, write_result):
        import warnings

        # Detect which interface we're using
        if hasattr(write_result, 'write_returns'):
            # New Ray interface (>= 2.38)
            write_returns = write_result.write_returns
            is_new_interface = True
        else:
            # Old Ray interface (< 2.38) - write_result is List[Block]
            write_returns = write_result
            is_new_interface = False

        if not write_returns:
            # No data was written, skip commit
            return

        fragments = []
        schema = None
        
        try:
            if is_new_interface:
                # New format: write_returns is List[List[Tuple[FragmentMetadata, pa.Schema]]]
                for write_return in write_returns:
                    if write_return:  # Check if not None/empty
                        for item in write_return:
                            if isinstance(item, tuple) and len(item) == 2:
                                fragment, schema_obj = item
                                fragments.append(fragment)
                                schema = schema_obj
            else:
                # Old format: nested lists
                for batch in write_returns:
                    if batch:  # Check if not None/empty
                        for item in batch:
                            if isinstance(item, tuple) and len(item) == 2:
                                fragment, schema_obj = item
                                fragments.append(fragment)
                                schema = schema_obj
        except Exception as e:
            warnings.warn(
                f"Failed to process write results: {e}. "
                "This might be due to Ray version compatibility issues.",
                DeprecationWarning,
            )
            return
        
        # Skip commit when there are no fragments
        if not schema or not fragments:
            return
            
        if self.mode in set(["create", "overwrite"]):
            op = lance.LanceOperation.Overwrite(schema, fragments)
        elif self.mode == "append":
            op = lance.LanceOperation.Append(fragments)
        
        lance.LanceDataset.commit(
            self.uri,
            op,
            read_version=self.read_version,
            storage_options=self.storage_options,
        )


class LanceDatasink(_BaseLanceDatasink):
    """Lance Ray Datasink.

    Write a Ray dataset to lance.

    If we expect to write larger-than-memory files,
    we can use `LanceFragmentWriter` and `LanceCommitter`.

    Parameters
    ----------
    uri : str
        The base URI of the dataset.
    schema : pyarrow.Schema
        The schema of the dataset.
    mode : str, optional
        The write mode. Default is 'append'.
        Choices are 'append', 'create', 'overwrite'.
    max_rows_per_file : int, optional
        The maximum number of rows per file. Default is 1024 * 1024.
    data_storage_version: optional, str, default None
        The version of the data storage format to use. Newer versions are more
        efficient but require newer versions of lance to read.  The default is
        "legacy" which will use the legacy v1 version.  See the user guide
        for more details.
    use_legacy_format : optional, bool, default None
        Deprecated method for setting the data storage version. Use the
        `data_storage_version` parameter instead.
    storage_options : Dict[str, Any], optional
        The storage options for the writer. Default is None.
    """

    NAME = "Lance"

    def __init__(
        self,
        uri: str,
        schema: Optional[pa.Schema] = None,
        mode: Literal["create", "append", "overwrite"] = "create",
        max_rows_per_file: int = 1024 * 1024,
        data_storage_version: Optional[str] = None,
        use_legacy_format: Optional[bool] = None,
        storage_options: Optional[Dict[str, Any]] = None,
        *args,
        **kwargs,
    ):
        super().__init__(
            uri,
            schema=schema,
            mode=mode,
            storage_options=storage_options,
            *args,
            **kwargs,
        )

        if use_legacy_format is not None:
            import warnings

            warnings.warn(
                "The `use_legacy_format` parameter is deprecated. Use the "
                "`data_storage_version` parameter instead.",
                DeprecationWarning,
            )

            if use_legacy_format:
                data_storage_version = "legacy"
            else:
                data_storage_version = "stable"

        self.max_rows_per_file = max_rows_per_file
        self.data_storage_version = data_storage_version
        # if mode is append, read_version is read from existing dataset.
        self.read_version: int | None = None

    @property
    def supports_distributed_writes(self) -> bool:
        return True

    @property
    def num_rows_per_write(self) -> int:
        return self.max_rows_per_file

    def get_name(self) -> str:
        return self.NAME

    def write(
        self,
        blocks: Iterable[Union[pa.Table, "pd.DataFrame"]],
        _ctx,
    ):
        return _write_fragment(
            blocks,
            self.uri,
            schema=self.schema,
            max_rows_per_file=self.max_rows_per_file,
            data_storage_version=self.data_storage_version,
            storage_options=self.storage_options,
        )


class LanceFragmentWriter:
    """Write a fragment to one of Lance fragment.

    This Writer can be used in case to write large-than-memory data to lance,
    in distributed fashion.

    Parameters
    ----------
    uri : str
        The base URI of the dataset.
    transform : Callable[[pa.Table], Union[pa.Table, Generator]], optional
        A callable to transform the input batch. Default is None.
    schema : pyarrow.Schema, optional
        The schema of the dataset.
    max_rows_per_file : int, optional
        The maximum number of rows per file. Default is 1024 * 1024.
    max_bytes_per_file : int, optional
        The maximum number of bytes per file. Default is 90GB.
    max_rows_per_group : int, optional
        The maximum number of rows per group. Default is 1024.
        Only useful for v1 writer.
    data_storage_version: optional, str, default None
        The version of the data storage format to use. Newer versions are more
        efficient but require newer versions of lance to read.  The default
        (None) will use the 2.0 version.  See the user guide for more details.
    use_legacy_format : optional, bool, default None
        Deprecated method for setting the data storage version. Use the
        `data_storage_version` parameter instead.
    storage_options : Dict[str, Any], optional
        The storage options for the writer. Default is None.

    """

    def __init__(
        self,
        uri: str,
        *,
        transform: Optional[Callable[[pa.Table], Union[pa.Table, Generator]]] = None,
        schema: Optional[pa.Schema] = None,
        max_rows_per_file: int = 1024 * 1024,
        max_bytes_per_file: Optional[int] = None,
        max_rows_per_group: Optional[int] = None,  # Only useful for v1 writer.
        data_storage_version: Optional[str] = None,
        use_legacy_format: Optional[bool] = False,
        storage_options: Optional[Dict[str, Any]] = None,
    ):
        if use_legacy_format is not None:
            import warnings

            warnings.warn(
                "The `use_legacy_format` parameter is deprecated. Use the "
                "`data_storage_version` parameter instead.",
                DeprecationWarning,
            )

            if use_legacy_format:
                data_storage_version = "legacy"
            else:
                data_storage_version = "stable"

        self.uri = uri
        self.schema = schema
        self.transform = transform if transform is not None else lambda x: x

        self.max_rows_per_group = max_rows_per_group
        self.max_rows_per_file = max_rows_per_file
        self.max_bytes_per_file = max_bytes_per_file
        self.data_storage_version = data_storage_version
        self.storage_options = storage_options

    def __call__(self, batch: Union[pa.Table, "pd.DataFrame"]) -> Dict[str, Any]:
        """Write a Batch to the Lance fragment."""

        transformed = self.transform(batch)
        if not isinstance(transformed, Generator):
            transformed = (t for t in [transformed])

        fragments = _write_fragment(
            transformed,
            self.uri,
            schema=self.schema,
            max_rows_per_file=self.max_rows_per_file,
            max_rows_per_group=self.max_rows_per_group,
            data_storage_version=self.data_storage_version,
            storage_options=self.storage_options,
        )
        
        # Handle empty results
        if not fragments:
            return {"fragment": np.array([]), "schema": np.array([])}
        # Serialize objects for Ray compatibility
        return {
            "fragment": np.array([fragment for fragment, _ in fragments], dtype=object),
            "schema": np.array([schema for _, schema in fragments], dtype=object),
        }
        
class LanceCommitter(_BaseLanceDatasink):
    """Lance Committer as Ray Datasink.

    This is used with `LanceFragmentWriter` to write large-than-memory data to
    lance file.
    """

    @property
    def num_rows_per_write(self) -> int:
        return 1

    def get_name(self) -> str:
        return f"LanceCommitter({self.mode})"

    def write(
        self,
        blocks: Iterable[Union[pa.Table, "pd.DataFrame"]],
        _ctx,
    ):
        """Passthrough the fragments to commit phase"""
        v = []
        for block in blocks:
            # If block is a dict (batch_format="default")
            if isinstance(block, dict):
                results = block.get("results", [])
                for item in results:
                    v.append((item["fragment"], item["schema"]))
            # If block is a PyArrow Table (sometimes Ray may pass this)
            elif isinstance(block, pa.Table):
                if "results" in block.schema.names:
                    # Each row is a dict: {"fragment": ..., "schema": ...}
                    for item in block["results"].to_pylist():
                        v.append((item["fragment"], item["schema"]))
            else:
                continue
        return v

def write_lance(
    data: ray.data.Dataset,
    output_uri: str,
    *,
    schema: Optional[pa.Schema] = None,
    mode: Literal["create", "append", "overwrite"] = "create",
    transform: Optional[
        Callable[[pa.Table], Union[pa.Table, Generator[None, pa.Table, None]]]
    ] = None,
    max_rows_per_file: int = 1024 * 1024,
    max_bytes_per_file: Optional[int] = None,
    storage_options: Optional[Dict[str, Any]] = None,
    data_storage_version: Optional[str] = None,
) -> None:
    """Write Ray dataset at scale.

    This method wraps the `LanceFragmentWriter` and `LanceCommitter` to write
    large-than-memory ray data to lance files.

    Parameters
    ----------
    data : ray.data.Dataset
        The dataset to write.
    output_uri : str
        The output dataset URI.
    transform : Callable[[pa.Table], Union[pa.Table, Generator]], optional
        A callable to transform the input batch. Default is identity function.
    schema : pyarrow.Schema, optional
        If provided, the schema of the dataset. Otherwise, it will be inferred.
    max_rows_per_file: int, optional
        The maximum number of rows per file. Default is 1024 * 1024.
    max_bytes_per_file: int, optional
        The maximum number of bytes per file. Default is 90GB.
    storage_options : Dict[str, Any], optional
        The storage options for the writer. Default is None.
    data_storage_version: optional, str, default None
        The version of the data storage format to use. Newer versions are more
        efficient but require newer versions of lance to read.  The default
        (None) will use the 2.0 version.  See the user guide for more details.
    """
    data.map_batches(
        LanceFragmentWriter(
            output_uri,
            schema=schema,
            transform=transform,
            max_rows_per_file=max_rows_per_file,
            max_bytes_per_file=max_bytes_per_file,
            storage_options=storage_options,
            data_storage_version=data_storage_version,
        ),
        batch_size=max_rows_per_file,
    ).write_datasink(
        LanceCommitter(
            output_uri, schema=schema, mode=mode, storage_options=storage_options
        )
    )

def _register_hooks():
    """Register lance hook to Ray for better integration.

    You can use `ray.data.Dataset.write_lance` to write Ray dataset to lance.
    Example:

    ```python
    import ray
    import lance
    from lance.ray.sink import _register_hooks

    _register_hooks()

    ray.data.range(10)
        .map(lambda x: {"id": x["id"], "str": f"str-{x['id']}"})
        .write_lance("~/data.lance")
    ```
    """
    ray.data.Dataset.write_lance = write_lance
