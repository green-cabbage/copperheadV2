"""Exact data-event deduplication for opt-in XZZ compaction."""
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

KEY = ('run', 'luminosityBlock', 'event')


def data_duplicate_rows(file_infos, raw_files):
    """Return row exclusions, preferring SingleMuon over the electron stream.

    Only selected Stage1 records participate. No event identifiers are converted
    through floating point. Equal keys in MC must never use this function.
    """
    source = {}
    for raw in raw_files:
        if '/SingleMuon/' in raw:
            priority = 0
        elif '/EGamma/' in raw or '/SingleElectron/' in raw:
            priority = 1
        else:
            raise ValueError(f'Unsupported data stream: {raw}')
        stem = Path(raw).stem
        if stem in source:
            raise ValueError(f'Ambiguous raw file stem: {stem}')
        source[stem] = (priority, raw)

    def origin(path):
        name = Path(path).stem
        if not name.startswith('part_'):
            raise ValueError(f'Cannot recover source of Stage1 shard: {path}')
        return source[name[5:].rsplit('_', 2)[0]]

    ordered = sorted(file_infos, key=lambda x: (origin(x[0])[0], str(x[0])))
    offsets = np.concatenate(([0], np.cumsum([n for _, n in ordered], dtype=np.int64)))
    keys = np.empty(int(offsets[-1]), dtype=[(k, 'u8') for k in KEY])

    def read(item):
        i, (path, count) = item
        a = pq.ParquetFile(path).read(columns=list(KEY))
        if len(a) != count:
            raise ValueError(f'Row count changed while compacting: {path}')
        for k in KEY:
            col = a[k].combine_chunks()
            if col.null_count:
                raise ValueError(f'Missing event identifier in {path}')
            values = col.to_numpy()
            if values.dtype.kind not in 'iu':
                raise ValueError(f'Non-integer event identifier in {path}')
            keys[k][offsets[i]:offsets[i + 1]] = values

    with ThreadPoolExecutor(max_workers=8) as pool:
        list(pool.map(read, enumerate(ordered)))
    unique, first = np.unique(keys, return_index=True)
    keep = np.zeros(len(keys), dtype=bool)
    keep[first] = True
    dropped = np.flatnonzero(~keep)
    excluded = {}
    for i in np.unique(np.searchsorted(offsets[1:], dropped, side='right')):
        rows = dropped[(dropped >= offsets[i]) & (dropped < offsets[i + 1])] - offsets[i]
        excluded[str(ordered[i][0])] = rows.tolist()

    def location(index):
        i = int(np.searchsorted(offsets[1:], index, side='right'))
        return {'parquet': str(ordered[i][0]), 'row': int(index - offsets[i]),
                'raw_file': origin(ordered[i][0])[1]}

    examples = []
    for index in dropped[:10]:
        key = keys[index]
        retained = first[np.searchsorted(unique, key)]
        examples.append({**{k: int(key[k]) for k in KEY},
                         'kept': location(retained), 'removed': location(index)})
    report = {'input_rows': len(keys), 'output_rows': len(unique),
              'duplicate_rows_removed': len(dropped),
              'policy': 'Exact run/luminosityBlock/event; SingleMuon first, then electron stream; lexical shard order within stream.',
              'examples': examples}
    return excluded, report
