# Reikia sugalvoti prompta, kad padetu labai tiksliai rusiuoti naratyvus

def find_components(embs_dict, items, get_shortest=True, print_single_row=False, reverse=False, clique=True, threshold=0.85, step=0.01, components_only=False, k=None, cc_view=None, iii=0):
    components=[]
    component_check_iter = 0
    while len(components) == 0 or k is not None and len(components) != k:
        if k is not None and k > len(items):
            break
        if len(components) != 0:
            component_check_iter+=1
            if component_check_iter>10 and len(components) >= k:
                break
            if len(components) > k:
                threshold-=step
                step /= 2
                threshold+=step
                print("step halved")
            else:
                threshold+=step
        embs_computed_calc = np.array(list({k: embs_dict[k] for k in items}.values())) @ np.array(list({k: embs_dict[k] for k in items}.values())).T
        print("embs_computed std", np.std(embs_computed_calc))
        # time.sleep(5)
        # if cc_view is None:
        #     cc_view = list(range(len(embs)))
        # embs_computed_calc = cp.asarray(embs[cc_view]) @ cp.asarray(embs[cc_view].T)

        # embs_computed_calc = embs_computed.copy()
        embs_computed_calc[embs_computed_calc < threshold] = 0
        embs_computed_calc[embs_computed_calc > threshold-0.1] = 1

        graph = nx.from_numpy_array(embs_computed_calc)
        del embs_computed_calc
        # gc.collect()
        print("finished_emb")
        # time.sleep(20)

        numbers = []
        if clique:
            now = time.time()
            components = list(nx.find_cliques(graph))
            print("cliques in", time.time()-now)
            components.sort(key=len, reverse=reverse)
            # print(type(components))
            # print(len(components))
            # print(components)
            for cc in components:
                if len(cc)>1:
                    numbers += cc
            
            is_already = set()
            for i1 in range(len(components)):
                to_remove = []
                for ii in components[i1]:
                    if ii in is_already:
                        # to_remove.append([i1, ii])
                        to_remove.append(ii)
                        # print("k",[i1, ii])
                    else:
                        is_already.add(ii)
                
                if len(to_remove) != 0:
                    # print(len(to_remove)-len(components[i1])>=-1, to_remove, components[i1])
                    for tr in to_remove:
                        components[i1].remove(tr)
            
            # for tr in to_remove:
            #     components
                
        else:
            components = list(nx.connected_components(graph))
            # print(type(components))
            print(len(components))
            # print(components)
            for cc in components:
                if len(cc)>1:
                    numbers += list(cc)
    
    print(len(numbers), len(set(numbers)))

    
    good_to_bad_mapping = {}
    sorted_components = {}
    bad_cids = set()

    # iii=0
    for cc in components:
        if k is not None or len(cc)>1:
            # print(print("doing comp"))
            # if len(cc) == len(items) or len(cc)>50 or k is not None and len(cc) < k:
            if len(cc) == len(items) or len(cc)>50:
                print("redoing", len(cc))
                to_add = find_components(embs_dict, [items[c] for c in cc], get_shortest=get_shortest, reverse=reverse, clique=clique, threshold=threshold+step, step=step, components_only=components_only, k=k)#, cc_view=list(copy.deepcopy(cc)), iii=iii)
                print()
                sorted_components |= {f"c_{iiii}": v for iiii, v in zip(range(iii, iii+len(to_add)), to_add.values())}
                iii+=len(to_add)

                print("finished redoing", len(cc), "added", )
                
                continue
            bad_cids.add(f"c_{iii}")
            cc = [items[c] for c in sorted(cc, key=lambda x: len(items[x]), reverse=not get_shortest)]
            # print([len(c) for c in cc])
            # print(type(cc), cc)
            if print_single_row:
                print(f"c_{iii}",len(cc), ", ".join(cc))
            else:
                print(f"c_{iii}",len(cc))
                for s in cc:
                    print(s)
                print()
            if components_only:
                sorted_components[f"c_{iii}"] = cc
            else:
                good_to_bad_mapping[cc[0]] = cc[1:]

            iii+=1
    
    print("finished function, threshold", threshold)
    if components_only:
        return sorted_components
    
    bad_to_good_mapping = {}
    for k, v in good_to_bad_mapping.items():
        for val in v:
            bad_to_good_mapping[val] = k

    return good_to_bad_mapping, bad_to_good_mapping







def find_identical(embs, stuff_to_join, get_shortest=True, print_single_row=False, reverse=True, intersectt=True, custom_threshold=None, components_only=True):
    embs_computed = embs @ embs.T

    sorted_embs_inds = np.argsort(embs_computed, axis=1)[:,::-1][:,:40]

    sorted_embs = embs_computed[np.arange(embs_computed.shape[0])[:, None], sorted_embs_inds]

    if custom_threshold is None:
        sorted_embs_diff = -np.diff(sorted_embs[:,:], axis=1)
        sorted_embs_answ = np.argmax(sorted_embs_diff, axis=1)

    else:
        sorted_embs_answ = np.argmax(sorted_embs < custom_threshold, axis=1)

    embs_to_join = np.argwhere(sorted_embs_answ > 0)

    potential_identical = {}
    # potential_identical_inds = {}
    # item_to_ind = {}

    for i in embs_to_join:
        i = int(i)

        inds = set()
        inds.add(i)
        # print(stuff_to_join[i])
        
        to_print = []

        j = sorted_embs_answ[i]
        if j > 0:
            potential_identical[stuff_to_join[i]] = set([stuff_to_join[i]])
            # potential_identical_inds[i] = set([i])
            # item_to_ind[stuff_to_join[i]] = i
            for jj in range(j+1):
                # if sorted_embs[i, jj] > 0.8:
                # print(i, jj)
                    ind = sorted_embs_inds[i,jj]
                    inds.add(int(ind))
                    potential_identical[stuff_to_join[i]].add(stuff_to_join[ind])
                    # potential_identical_inds[i].add(int(ind))
                    # item_to_ind[stuff_to_join[ind]] = int(ind)
                    # if print_single_row:
                    #     to_print.append(stuff_to_join[ind])
                    # else:
                    #     print(sorted_embs[i][:sorted_embs_answ[i]+2],stuff_to_join[ind])
                        # print(stuff_to_join[ind])

            # if print_single_row:
            #     # print(inds, ", ".join(to_print))
            #     print(len(inds), ", ".join(potential_identical[stuff_to_join[i]]))
            # else:
            #     print()
        
        # print(len(potential_identical))
    print("-----------------------------------------------------------------")

    checked = set()
    to_join = []
    for k, v in potential_identical.items():
        if k in checked:
            continue

        checked.add(k)

        this_join = copy.deepcopy(v)
        for item in v:
            # print(item_to_ind[item])
            # print(embs_computed[item_to_ind[k],item_to_ind[item]], item_to_ind[k],item_to_ind[item])
            # print(potential_identical_inds[item_to_ind[item]])
            # print(sorted_embs[potential_identical_inds[item_to_ind[item]]])
            if item in potential_identical:
                # print("do", item)
                if intersectt:
                    this_join &= potential_identical[item]
                else:
                    this_join |= potential_identical[item]
                checked.add(item)
        
        if len(this_join) != 0:
            to_join.append(this_join)

        # print()
    
    for stuffs in to_join:
        if print_single_row:
            print(len(stuffs), ", ".join(stuffs))
        else:
            print(len(stuffs))
            for s in stuffs:
                print(s)
            print()
    
    if components_only:
        return to_join
    
    good_to_bad_mapping = {}
    for tj in to_join:
        key = min(tj) if get_shortest else max(tj)
        good_to_bad_mapping[key] = list(tj - set([key]))
    
    bad_to_good_mapping = {}
    for k, v in good_to_bad_mapping.items():
        for val in v:
            bad_to_good_mapping[val] = k
    
    return good_to_bad_mapping, bad_to_good_mapping