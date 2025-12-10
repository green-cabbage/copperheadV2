from coffea.dataset_tools import rucio_utils
from coffea.dataset_tools.preprocess import preprocess
from coffea.nanoevents import NanoEventsFactory, NanoAODSchema, BaseSchema
import json
import coffea
from distributed import LocalCluster, Client
import awkward as ak
import dask

def has_overlap(z1, z2, fields2compare):
    """
    Return True if any (run, luminosityBlock, event) tuple
    appears in both awkward zips z1 and z2.
    """


    # z1 = ak.zip({field : z1[field] for field in fields2compare}).compute()
    # z2 = ak.zip({field : z2[field] for field in fields2compare}).compute()

    z1 = ak.zip({field : z1[field] for field in fields2compare})
    z2 = ak.zip({field : z2[field] for field in fields2compare})
    
    z1, z2 = dask.compute(z1, z2)
    print(z1)
    print(z2)
    # Build Python sets of (run, lumi, event) triplets
    # Convert the selected fields into tuple keys

    keys1 = set(
        zip(*(ak.to_numpy(z1[f]) for f in fields2compare))
    )
    keys2 = set(
        zip(*(ak.to_numpy(z2[f]) for f in fields2compare))
    )
    # Check if intersection is non-empty
    # print(keys1 & keys2)
    n_overlap = len(keys1 & keys2)
    print(f"overlapped events: {(keys1 & keys2)}")
    print(f"number of overlapped events: {n_overlap}")
    return n_overlap > 0
    # return keys1, keys2

def getNanoEventFromDasQuery(das_query: str, allowlist_sites=["T2_US_Purdue"]) -> coffea.nanoevents:
    rucio_client = rucio_utils.get_rucio_client() 
    
    outlist, outtree = rucio_utils.query_dataset(
        das_query,
        client=rucio_client,
        tree=True,
        scope="cms",
    )
    
    outfiles,outsites,sites_counts = rucio_utils.get_dataset_files_replicas(
        outlist[0],
        allowlist_sites=allowlist_sites,
        mode="full",
        client=rucio_client,
        # partial_allowed=True
    )
    fnames = [file[0] for file in outfiles if file != []]
    file_input = {fname : {"object_path": "Events"} for fname in fnames}
    events = NanoEventsFactory.from_root(
            file_input,
            metadata={},
            schemaclass=NanoAODSchema,
            uproot_options={"timeout":4*2400},
    ).events()
    # print(file_input)
    # final_output = {
    #     "data" :{"files" :file_input}
    # }
    # step_size = 100_000

    # val = "Events"
    # file_dict = {}
    # for file in fnames:
    #     file_dict[file] = val
    # final_output = {
    #     "data" :{"files" :file_dict}
    # }
    # files_available, files_total = preprocess(
    #     final_output,
    #     step_size=step_size,
    #     align_clusters=False,
    # )

    # print(files_available)
    # events = NanoEventsFactory.from_root(
    #         files_available,
    #         metadata={},
    #         schemaclass=NanoAODSchema,
    #         uproot_options={"timeout":4*2400},
    # ).events()
    return events
    
if __name__ == "__main__":
    client = Client(n_workers=60,  threads_per_worker=1, processes=True, memory_limit='30 GiB')
    doubleMu_das_query="/DoubleMuon/Run2017B-UL2017_NanoAODv15-v1/NANOAOD"
    events_doubleMu = getNanoEventFromDasQuery(doubleMu_das_query)
    singleMu_das_query="/SingleMuon/Run2017B-UL2017_NanoAODv15-v1/NANOAOD"
    events_singleMu = getNanoEventFromDasQuery(singleMu_das_query)

    fields2compare = ["run", "luminosityBlock", "event"]
    _ = has_overlap(events_doubleMu, events_singleMu, fields2compare)  # -> True, since (2, 12, 200) is common
    