import imports

# butinai butinai pagal data rusiuojam

query = """
SELECT * FROM `acrux-440014.shared_views.telegram_post_views` 
WHERE TIMESTAMP_TRUNC(uploaded_at, MONTH) between "2024-01-01" and "2024-03-27"
AND LENGTH(message) > 100
order by uploaded_at asc;
"""
df2 = client.query(query).to_dataframe()
print(len(df2))


# viska irasom pas save i atminti
for i in range(len(df2)//1000000+1):
    with open(f"telegram_2024_01_sorted/telegram_2024_01_{i}.pkl", "wb") as f:
        pickle.dump(df2.iloc[i*1000000:i*1000000+1000000], f)


# viska taip darom, kad taupytumeme ram, pas mane kompe 32gb ram, bet kaisu 10 min postu dirbam viskas keiciasi

# for ...
#   atidarome issaugotus atmintyje dataframus
#       df = data_cleaner(df)
#   issaugome isvalytus dataframus ant virsaus

# sita zingsni galim praleist

# atskiriam pagal "uploaded_at" timestampu density,
# tam, kad veliau UMAP darytume tais laikotarpiais kai buvo daug tekstu
# kitaip tariant darom atskirtis, jeigu ta diena postu buvo maziausiai, tada atskiriam
final_iter=0
for i in list(range(9))[:]:
    print("start",i)
    # lokacija telegram lenteliu. Cia assuminama, kad atsisiuntus is google cloud paskirstom i atskirus failus
    with open(f"telegram_2024_01_sorted/telegram_2024_01_{i}.pkl", "rb") as f:
        df_start = pickle.load(f)
    
    df_start = df_start[~df_start.index.duplicated(keep='last')]
    
    id_to_timestamps = df_start.astype({"uploaded_at":"int64"})["uploaded_at"].to_dict()
    id_to_timestamps_list = list(id_to_timestamps.values())
    counts, bins = np.histogram(id_to_timestamps_list, bins=1000)

    counts = scipy.ndimage.gaussian_filter1d(counts, 20)
    bins = np.array(np.linspace(0,1, len(counts)+1))

    minima, properties = scipy.signal.find_peaks(-counts)

    bins_indexes = [0]+[int(len(id_to_timestamps_list)*bins[i]) for i in minima]+[len(id_to_timestamps_list)]

    bins_indexes_diff = np.diff(bins_indexes)
    done_something=False
    deleted=0
    for iw in np.where(bins_indexes_diff < 50000)[0]:
        if bins_indexes_diff[iw]<50000:
            print("removed", bins_indexes[iw+1-deleted])
            del bins_indexes[iw+1-deleted]
            deleted+=1
            done_something=True

    bins_indexes_diff = np.diff(bins_indexes)
    for iw in np.where(bins_indexes_diff > 150000)[0]:
        divvv = bins_indexes_diff[iw]//150000+1
        divf = bins_indexes_diff[iw]//divvv
        print(divf)
        for ii in range(divvv-1):
            bins_indexes.append(bins_indexes[iw+1]-divf*(ii+1))
            print("added", bins_indexes[iw+1]-divf*(ii+1), divf*(ii+1))
            done_something=True
    
    bins_indexes = sorted(bins_indexes)

    if done_something:
        print(np.diff(bins_indexes))

    for i1, i2 in zip(bins_indexes[:-1], bins_indexes[1:]):
        df = df_start.iloc[i1:i2]
    
        with open(f"telegram_2024_01_sorted/new/telegram_2024_01_{final_iter}.pkl", "wb") as f:
            pickle.dump(df, f)
        
        print("done",final_iter, len(df))
        final_iter+=1



# nebutinas, bet patartinas zingsnis

# embedingus rasom i faila, ne i kietaji diska, ir storinam pointerius embedingu nuo pradzios iki galo
# veliau viska labai pagreitins.
# irasinet viska i sql duombaze ilgai uztruks

telegram_embs_full_dict = {}

with open("telegram_2024_01/telegram_2024_01_embs_full.txt", "ab+") as f_emb:
    for i in list(range(105))[:]:
        # loadinam messagus
        with open(f"telegram_2024_01/telegram_2024_01_{i}.pkl", "rb") as f:
            dft1 = pickle.load(f)
        
        #
        # cia rasom koda kad embedint visus messages, jeigu embeddingu nepridejom issaugotame dataframe
        #
        
        print("loaded", i)
        # written_cnt = 0
        telegram_embs_full_temp = {}
        try:
            with open("telegram_2024_01/telegram_2024_01_embs_full.txt", "ab+") as f_emb:
                for index, emb in dft1["embeddings"].to_dict().items():
                    if index in telegram_embs_full_dict:
                        continue
                    start = f_emb.tell()
                    end = f_emb.write(pickle.dumps(emb))
                    telegram_embs_full_temp[index] = {"start":start, "end":end, "good":True}
                    # written_cnt+=1
        except Exception as e:
            print(i, e)
            continue
        
        telegram_embs_full_dict |= telegram_embs_full_temp
        
        print("finished", len(telegram_embs_full_temp))

with open("telegram_2024_01/telegram_2024_01_full_embs_indexes.pkl", "wb") as f:
    pickle.dump(telegram_embs_full_dict, f)



# jau pravalytus ir padalintus dataframus i batchus reducinam embeddingus
# embeddingus kraunam is kietojo disko
# reduced embeddings saugom prie dataframe, nes sumazintos dimensijos nedaug uzema
with open("/data/telegram_2024_01/telegram_2024_01_embs_full.txt", "rb") as f_emb:
    # for index, info in telegram_embs_full_dict.items():
    # with open()
    for i in list(range(73))[8:]:
        print("start",i)
        with open(f"/data/telegram_2024_01_sorted/new/telegram_2024_01_{i}.pkl", "rb") as f:
            df = pickle.load(f)

        df.index = df.index.astype("int64")
        df = df[~df.index.duplicated(keep='last')]
        now = time.time()

        print("start", len(df))
        # break
        embs_dict = {}
        for index in df.index:
            index = int(index)
            # print(index)
            # if index in telegram_embs_full_dict_set:
            tracker = telegram_embs_full_dict.get(index)
            if tracker:
                f_emb.seek(telegram_embs_full_dict[index]["start"])
                # embs_dict[index] = pickle.loads(f_emb.read(tracker["end"]))
                embs_dict[index] = f_emb.read(tracker["end"])
        
        for index in embs_dict:
            embs_dict[index] = pickle.loads(embs_dict[index])
        
        print("finished getting embs", len(embs_dict), time.time()-now)

        df = df.loc[list(embs_dict)]

        reduced_umap = reduce_umap(np.array(list(embs_dict.values())), 25, 10000, 20)
        df["reduced_embeddings"] = reduced_umap.tolist()

        with open(f"/data/telegram_2024_01_sorted/new/telegram_2024_01_{i}.pkl", "wb") as f:
            pickle.dump(df, f)



# dabar darom naratyvu traukima (sintetiniu textu reprezentacijas, kurias galima embeddint ir perskirstyt viska tiksliau)

# file_names = ["/data/telegram_large_df_processed.pkl"] + [f"/data/telegram_new_05_15_full{i+1}.pkl" for i in list(range(73))[60:]]
file_names = [f"/data/telegram_2024_01_sorted/new/telegram_2024_01_{i+1}.pkl" for i in list(range(72))[60:]]
# file_names = ["/data/telegram_large_df_processed.pkl"]
# file_names

import gc

# async def process_messages(file_names):
    # all_topics_and_entities = []
    # all_cluster_to_text_ids = []
all_cluster_to_text_ids = {}
all_topics_and_entities = {}
all_bad_cids = {}

# clustering index
ci = 0
for fn in file_names:
    # print(fn)
    # continue
    with open(fn, "rb") as f:
        data_df = pickle.load(f)

    text_ids = data_df.index.to_list()
    orig_emb = np.array(data_df["reduced_embeddings"].to_list())

    cluster_to_text_ids, cluster_text_inds, labels = hdbscan_cluster(orig_emb, text_ids, mcs=20, rt=False)
    print("start summaries", fn, "clusters:", len(cluster_to_text_ids))

    # time.sleep(1)
    
    topics_and_entities, bad_cids = await get_summaries_from_clusters(dict(list(cluster_to_text_ids.items())[:]), data_df["message"].to_dict(), print_token_usage=True, sample_from_centers=False, data_df=None, static_random=False)
    # all_cluster_to_text_ids.append(cluster_to_text_ids)
    # all_topics_and_entities.append(topics_and_entities)
    print(type(topics_and_entities))
    all_cluster_to_text_ids |= {f"{ci}_{c}": v for c, v in cluster_to_text_ids.items()}
    all_topics_and_entities |= {f"{ci}_{c}": v for c, v in topics_and_entities.items()}
    all_bad_cids |= {f"{ci}_{c}": v for c, v in bad_cids.items()}
    ci+=1

# galim patikrint llm isvedamas statistikas, kiek is viso klasteriu yra ir kiek sintetiniu naratyvu
print(sum([1 if v["input_cluster_processing_recommendation"]["action_needed_before_rst_generation_is_optimal"] != "None" else 0 for c, v in all_topics_and_entities.items()]))
clusters_to_redo = {c: v for c, v in all_topics_and_entities.items() if v["input_cluster_processing_recommendation"]["action_needed_before_rst_generation_is_optimal"] != "None"}
print()
# print(sum([1 if v["cluster_cohesion_and_granularity"]["cohesion_level"] == "High" else 0 for c, v in all_topics_and_entities.items()]))
# print(sum([1 if v["cluster_cohesion_and_granularity"]["cohesion_level"] == "Medium" else 0 for c, v in all_topics_and_entities.items()]))
# print(sum([1 if v["cluster_cohesion_and_granularity"]["cohesion_level"] == "Low" else 0 for c, v in all_topics_and_entities.items()]))
print(sum([1 if v["internal_analysis_for_rst_generation"]["coherence_for_rst_creation"] == "Sufficient" else 0 for c, v in all_topics_and_entities.items()]))
print(sum([1 if v["internal_analysis_for_rst_generation"]["coherence_for_rst_creation"] == "Borderline" else 0 for c, v in all_topics_and_entities.items()]))
print(sum([1 if v["internal_analysis_for_rst_generation"]["coherence_for_rst_creation"] == "Insufficient" else 0 for c, v in all_topics_and_entities.items()]))
print()
print(sum([len(v["representative_synthetic_texts_for_embedding"]) for c, v in all_topics_and_entities.items()]))


# issaugom iskart viska i kietaji diska
with open("/data/telegram_2024_01_sorted/new/more_61_processed_package.pkl", "rb") as f:
    pickle.dump({
    "all_cluster_to_text_ids":all_cluster_to_text_ids,
    "all_topics_and_entities":all_topics_and_entities,
    "all_bad_cids":all_bad_cids,
}, f)
with open("/data/telegram_2024_01_sorted/new/more_61_processed_package.pkl", "rb") as f:
    loaded = pickle.load(f)
    all_cluster_to_text_ids = loaded["all_cluster_to_text_ids"]
    all_topics_and_entities = loaded["all_topics_and_entities"]
    all_bad_cids = loaded["all_bad_cids"]
    del loaded




# primary theme greiciausiai nenaudosim
primary_theme = [v["internal_analysis_for_rst_generation"]["main_subject_summary"] for c, v in all_topics_and_entities.items()]
primary_theme_to_synthetic_texts = {v["internal_analysis_for_rst_generation"]["main_subject_summary"]: [vv["synthetic_text_content"] for vv in v["representative_synthetic_texts_for_embedding"]] for c, v in all_topics_and_entities.items()}
synthetic_texts = [vv["synthetic_text_content"] for c, v in all_topics_and_entities.items() for vv in v["representative_synthetic_texts_for_embedding"]]
print(len(primary_theme))



synthetic_texts_embed_dict = await aembed_many_texts(synthetic_texts)
print(len(synthetic_texts_embed_dict))










# ieskom panasiu arba vienodu naratyvu (nebaigta)
synthetic_texts_components = find_components(synthetic_texts_embed_dict, synthetic_texts, get_shortest=False, clique=False, components_only=True, threshold=0.9, step=0.01)



prompt_end = """

evaluate how many unique narratives are in the text. For each narrative generate a synthetic text that only represents that narrative or those narratives. Also write the differences of facts (if there are), which you did not include in the narratives."""
# , highlight the differences.

# evaluate if we can join them or some or all of them should be kept separate"""

# are the narratives simmilar that we can create a synthetic text that will have """

to_join_clusters_test = await prompt_tester({k: v for k, v in synthetic_texts_components.items() if len(v)>5 and len(v)<10}, "", prompt_end)

