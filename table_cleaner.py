def table_cleaner(df):
    print(df.columns.to_list())
    try:
        df = df.drop(["shared_post_id", "message_id"], axis=1).dropna()
    except:
        print("couldn't drop shared_post_id, message_id")
    
    try:
        df["id"] = df["id"].astype("str")
    except:
        print("couldn't change index type")

    try:
        df = df.set_index("id")
    except:
        print("couldn't set_index")
    # df.index = df.index.astype("int64")

    print(df.columns.to_list())
    # df["message"].to_dict()
    # print("start")
    removed = url_removal(df["message"].to_dict())
    df["message"] = removed
    print("removed")

    df = df[df["message"].apply(lambda x: len(x)) > 100]
    print("applied")

    views = df["views_count"].to_list()
    reaction_count = df["reaction_count"].to_list()
    share_count = df["share_count"].to_list()
    # normalized_views = views/np.linalg.norm(views)
    normalized_views = np.sqrt(views/np.max(views))
    normalized_reaction_count = np.sqrt(reaction_count/np.max(reaction_count))
    normalized_share_count = np.sqrt(share_count/np.max(share_count))

    score = (normalized_views+normalized_reaction_count+normalized_share_count)/3

    df["score"] = score

    return df