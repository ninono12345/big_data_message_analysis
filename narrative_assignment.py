# lyginam tekstus su naratyvais

def compare_texts_to_narratives(text_id_to_emb, narr_to_emb, threshold=0.7, cpu=True):
    text_ids = list(text_id_to_emb.keys())

    calc_time = time.time()
    if cpu:
        text_to_nar_computed = np.array(list(text_id_to_emb.values())) @ np.array(list(narr_to_emb.values())).T
    else:
        text_to_nar_computed = cp.asnumpy( cp.asarray(np.array(list(text_id_to_emb.values()))) @ cp.asarray(np.array(list(narr_to_emb.values()))).T )
    print("calc", time.time()-calc_time)

    best_score = np.max(text_to_nar_computed, axis=1)
    best_narr_ind = np.argmax(text_to_nar_computed, axis=1)

    narratives = list(narr_to_emb.keys())
    bad_ids = set()
    text_ids_to_narratives = {}
    ids_to_scores = {}
    for i, bni in enumerate(best_narr_ind):
        if best_score[i] < threshold:
            bad_ids.add(text_ids[i])
            continue
        text_ids_to_narratives[text_ids[i]] = narratives[bni]
        ids_to_scores[text_ids[i]] = best_score[i]
    
    # narratives_to_ids = {}
    
    return text_ids_to_narratives, bad_ids, ids_to_scores

#

def compare_texts_to_narratives_large(telegram_embs_full_dict, text_ids, narr_to_emb, threshold=0.7, cpu=False, emb_file="/data/telegram_2024_01/telegram_2024_01_embs_full.txt", limit=100000):
    all_text_ids_to_narratives = {}
    all_bad_ids = set()
    all_ids_to_scores = {}

    with open(emb_file, "rb") as f_emb:
        for i in range(0, len(text_ids), limit):
            embs_dict = {}
            to_do_text_ids = text_ids[i:i+limit]
            for index in to_do_text_ids:
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
            print("got_embs", i)

            text_ids_to_narratives, bad_ids, ids_to_scores = compare_texts_to_narratives(embs_dict, narr_to_emb, threshold, cpu)

            all_text_ids_to_narratives |= text_ids_to_narratives
            all_bad_ids |= bad_ids
            all_ids_to_scores |= ids_to_scores

            print("finished",i)

    return all_text_ids_to_narratives, all_bad_ids, all_ids_to_scores