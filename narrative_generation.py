def get_n_samples_from_center(text_ids_list, data_df, n_samples):
    cembs = np.array(data_df["embeddings"][text_ids_list].to_list())
    cavg = np.average(cembs, axis=0)
    idx = np.sum(np.abs(cembs-cavg), axis=1).argsort()[:n_samples]

    return [text_ids_list[i] for i in idx]




# Reikia patobulint prompta. Bet sitas variantas irgi tinka
async def get_summaries_from_clusters(cluster_to_text_ids: dict[int, list[str]], original_texts: dict[str, str], print_token_usage=False, sample_from_centers=False, data_df=None, static_random=False):

    cluster_to_samples = {}
    inp_tokens = 0
    out_tokens = 0
    for cluster_idx in cluster_to_text_ids.keys():
        if cluster_idx == "-1":
            continue
        text_ids_list = cluster_to_text_ids[cluster_idx]
        to_get = int(len(text_ids_list)*0.2)
        to_get = 50 if to_get > 50 else to_get
        to_get = 10 if to_get < 10 else to_get
        n_samples = min(to_get, len(text_ids_list))
        if n_samples > 0:  # Only sample if there are texts
            if sample_from_centers:
                sampled_texts = get_n_samples_from_center(text_ids_list, data_df, n_samples)
            else:
                if static_random:
                    np.random.seed(0)
                sampled_indices = np.random.choice(
                    len(text_ids_list), 
                    size=n_samples, 
                    replace=False  # No duplicates
                )
                sampled_texts = [text_ids_list[i] for i in sampled_indices]
            cluster_to_samples[cluster_idx] = sampled_texts
        else:
            cluster_to_samples[cluster_idx] = []
    
    check_prompts = {}
    for cluster_idx, text_ids_list in cluster_to_samples.items():
        if len(text_ids_list) > 0:
            # print("til", len(text_ids_list))
            # format of input texts should be 1. 2. 3. etc.
            # if not get_themes_only:
                # formatted_texts = "\n".join([f"{i+1}. {emoji_pattern.sub(r'', original_texts[text_id])}" for i, text_id in enumerate(text_ids_list)])
                formatted_texts = "\n".join([f"{i+1}. {original_texts[text_id]}" for i, text_id in enumerate(text_ids_list)])


#                 check_prompts[cluster_idx] = f"""
# Analyze the following text samples, which have been grouped into a cluster based on their similarity:
# {formatted_texts}

# Your Task:
# Your primary goal is to understand what this cluster of texts is collectively about. You need to synthesize the information from ALL provided text samples to produce a consolidated analysis of the cluster. Do not focus on a single sample, but on the commonalities and overarching themes across all of them.

# The output should be useful for subsequent statistical analysis of a large corpus, meaning the fields you populate should be consistent and aggregable.

# Provide your response in this exact JSON format, directly following "Summary:":

# Summary:
# {{
#   "cluster_topic": "A concise and descriptive title for this cluster (3-7 words). Example: 'User Complaints about App Performance'",
#   "cluster_summary": "A 1-2 sentence, highly condensed summary that captures the central theme and essence of the texts in this cluster. This should synthesize the information, not just pick one text.",
#   "keywords": [
#     "A list of 5-7 single words or short phrases (2-3 words max) that are most representative of the cluster's content. These should be suitable for tagging."
#   ],
#   "dominant_sentiment": {{
#     "label": "Choose one: Positive, Negative, Neutral, or Mixed.",
#     "explanation": "If 'Mixed', briefly explain the conflicting sentiments. Otherwise, a short justification for the chosen label. Example: 'Users express frustration with frequent crashes.'"
#   }},
#   "dominant_intent": {{
#     "label": "Identify the most likely primary purpose or intent behind these texts. Examples: 'Seeking Information', 'Requesting Support', 'Providing Feedback (Positive)', 'Providing Feedback (Negative)', 'Expressing Frustration', 'Sharing Experience', 'Marketing/Promotion', 'General Discussion', 'Bug Report', 'Feature Request'. Be specific.",
#     "confidence": "Estimate your confidence in this intent label (Low, Medium, High)."
#   }},
#   "key_entities_or_concepts": [
#     "A list of 3-5 key named entities (like product names, company names, specific features, people, locations) or core concepts frequently mentioned and central to the discussion. These are often nouns or noun phrases."
#   ],
#   "common_phrases_or_patterns": [
#     "List 2-3 recurring short phrases or sentence structures (if any) that are characteristic of this cluster. This helps identify common ways of expression."
#   ],
#   "cluster_cohesion_assessment": {{
#     "level": "Rate the perceived cohesion: High (texts are very similar), Medium (texts share a clear theme but with some variation), Low (texts are loosely related).",
#     "justification": "Briefly explain your cohesion rating, noting any significant outliers or sub-themes if present."
#   }}
# }}
# """

# 5.  **Key Entities (if prominent):** List any frequently mentioned and important named entities (People, Organizations, Locations, Products, etc.). If none are prominent, state "None."
# "key_entities": {{
#     "people": ["string", ...],
#     "organizations": ["string", ...],
#     "locations": ["string", ...],
#     "products": ["string", ...],
#     "other": ["string", ...]
#   }},


#                 check_prompts[cluster_idx] = f"""
# Analyze the following text samples, which represent a cluster of texts grouped by similarity:
# {formatted_texts}

# Your Task:
# Your primary goal is to thoroughly analyze this cluster of texts and generate insightful outputs that can be used for both immediate understanding and for a downstream data processing pipeline. This pipeline will involve using highly accurate synthetic texts you create as new cluster centroids for the entire original corpus. The success of this downstream process hinges on the precision and faithfulness of these synthetic texts.

# Specifically, you need to:

# 1.  **Cluster Analysis:** Provide a comprehensive analysis of the given text samples. This includes identifying the central theme(s), key entities, sentiment, style, and an assessment of the cluster's internal coherence.
# 2.  **Representative Synthetic Texts (RSTs) - CRITICAL ACCURACY REQUIRED:**
#     *   Generate one or more "Representative Synthetic Texts." Each RST must be a **highly accurate and semantically precise** piece of text that **faithfully epitomizes** a distinct theme or sub-theme found within the provided samples.
#     *   **Crucially, the content of each RST must be directly supportable and derivable from the input texts.** Avoid introducing external concepts or creating overly broad generalizations that dilute its representational accuracy. The RST should feel like a "perfect, distilled essence" of the texts it represents.
#     *   These RSTs will be embedded using a transformer model. Texts from the original corpus will then be assigned to the RST that is semantically closest. The effectiveness of this new clustering entirely depends on the **accuracy and specificity** of these RSTs.
#     *   If the provided text samples cover multiple discernible sub-themes, you MUST generate a separate, **equally accurate and specific**, RST for each major sub-theme you identify. This is preferred over a single, potentially less accurate, synthetic text. Aim for 1-5 RSTs per input cluster, depending on its diversity.
# 3.  **Cohesion Assessment & Re-clustering Recommendation (for this input cluster):**
#     *   Assess the cohesion of the provided text samples.
#     *   If, and only if, the provided samples are so diverse or jumbled that you cannot generate a set of **specific, meaningful, and accurate** RSTs (even if multiple), then you should recommend that *this current input cluster sample be sub-divided further*. In this specific case, you would suggest how many sub-clusters (e.g., k for k-means) might be appropriate for these *input texts* before attempting to generate RSTs from those more coherent sub-groups. The goal is to ensure that RSTs can always be generated with high fidelity.
#     *   If you can generate good, accurate RSTs (even if multiple), then no re-clustering of the input is needed at this stage.

# Provide your response in this exact JSON format:

# {{
#   "overall_cluster_analysis": {{
#     "primary_theme_summary": "A concise 1-2 sentence summary describing the overarching main theme of this text cluster.",
#     "predominant_language": "e.g. lt, ru, en",
#     "key_topics_or_keywords": ["list", "of", "5-10", "key", "topics", "or", "keywords"],
#     "common_entities_or_concepts": ["list", "of", "recurring", "important", "named entities", "or", "technical concepts", "(in predominant language)"],
#     "perceived_text_style_and_purpose": "e.g., Formal technical documentation, Informal customer inquiries, Persuasive marketing copy, Legal disclaimers, News reporting, etc."
#   }},
#   "cluster_cohesion_and_granularity": {{
#     "cohesion_level": "High | Medium | Low",
#     "justification_for_cohesion": "Briefly explain why you chose this cohesion level (e.g., 'Texts discuss the same specific feature of a product' vs. 'Texts broadly relate to finance but cover diverse subtopics').",
#     "number_of_distinct_sub_themes_identified": "Integer (e.g., 1 if highly cohesive, 2-5 if moderately diverse with clear sub-themes)"
#   }},
#   "representative_synthetic_texts_for_embedding": [ // Always provide at least one. Provide more if distinct sub-themes are identified.
#     {{
#       "rst_id": "RST_1",
#       "synthetic_text_content": "A highly specific, concise, and **semantically accurate** synthetic text for the (first) identified theme/sub-theme (english). This text must be a faithful representation ideal for precise embedding.",
#       "represented_sub_theme_description": "A brief (1 sentence) description of the specific sub-theme or nuance this RST captures, highlighting its accurate representation of the source texts.",
#       "estimated_predominant_sentiment_label": "Positive | Negative | Neutral | Mixed | Objective"
#     }}
#     // Add more RST objects here if multiple distinct sub-themes were identified and warrant separate, equally accurate RSTs.
#     // e.g., {{ "rst_id": "RST_2", "synthetic_text_content": "...", "represented_sub_theme_description": "..." }}
#   ],
#   "input_cluster_processing_recommendation": {{ // This section is about the *current input texts* you just analyzed.
#     "action_needed_before_rst_generation_is_optimal": "None | Subdivide_Input_Cluster",
#     "justification_if_subdivide_needed": "Explain ONLY if 'Subdivide_Input_Cluster' is chosen. Detail why the current input is too diverse/jumbled to generate meaningful and **accurate** RSTs, even multiple ones. (e.g., 'The texts are too disparate, covering unrelated topics like X, Y, and Z, making specific and **accurate** RST generation ineffective.')",
#     "suggested_k_for_input_sub_clustering": "If 'Subdivide_Input_Cluster', suggest an integer (e.g., 2, 3, 4) for how many sub-clusters these specific input texts should be broken into for more focused analysis and subsequent **accurate** RST generation."
#   }}
# }}
# """






#                 check_prompts[cluster_idx] = f"""
# Analyze the following text samples, which represent a cluster of texts grouped by similarity:
# {formatted_texts}

# Your Task:
# Your primary goal is to thoroughly analyze this cluster of texts and generate insightful outputs that can be used for both immediate understanding and for a downstream data processing pipeline. This pipeline will involve using highly accurate synthetic texts you create as new cluster centroids for the entire original corpus. The success of this downstream process hinges on the precision and faithfulness of these synthetic texts.

# Specifically, you need to:

# 1.  **Cluster Analysis:** Provide a comprehensive analysis of the given text samples. This includes identifying the central theme(s), key entities, sentiment, style, and an assessment of the cluster's internal coherence.
# 2.  **Representative Synthetic Texts (RSTs) - CRITICAL ACCURACY REQUIRED:**
#     *   Generate one or more "Representative Synthetic Texts." Each RST must be a **highly accurate and semantically precise** piece of text that **faithfully epitomizes** a distinct theme or sub-theme found within the provided samples.
#     *   **Crucially, the content of each RST must be directly supportable and derivable from the input texts.** Avoid introducing external concepts or creating overly broad generalizations that dilute its representational accuracy. The RST should feel like a "perfect, distilled essence" of the texts it represents.
#     *   These RSTs will be embedded using a transformer model. Texts from the original corpus will then be assigned to the RST that is semantically closest. The effectiveness of this new clustering entirely depends on the **accuracy and specificity** of these RSTs.
#     *   If the provided text samples cover multiple discernible sub-themes, you MUST generate a separate, **equally accurate and specific**, RST for each major sub-theme you identify. This is preferred over a single, potentially less accurate, synthetic text. Aim for 1-5 RSTs per input cluster, depending on its diversity.
# 3.  **Cohesion Assessment & Re-clustering Recommendation (for this input cluster):**
#     *   Assess the cohesion of the provided text samples.
#     *   If, and only if, the provided samples are so diverse or jumbled that you cannot generate a set of **specific, meaningful, and accurate** RSTs (even if multiple), then you should recommend that *this current input cluster sample be sub-divided further*. In this specific case, you would suggest how many sub-clusters (e.g., k for k-means) might be appropriate for these *input texts* before attempting to generate RSTs from those more coherent sub-groups. The goal is to ensure that RSTs can always be generated with high fidelity.
#     *   If you can generate good, accurate RSTs (even if multiple), then no re-clustering of the input is needed at this stage.

# Provide your response in this exact JSON format:

# {{
#   "overall_cluster_analysis": {{
#     "primary_theme_summary": "A concise few sentence summary describing the overarching main theme of this text cluster.",
#     "predominant_language": "e.g. lt, ru, en",
#     "key_topics_or_keywords": ["list", "of", "5-10", "key", "topics", "or", "keywords"],
#     "common_entities_or_concepts": ["list", "of", "recurring", "important", "named entities", "or", "technical concepts", "(these should be listed in the original language of the texts)"],
#     "perceived_text_style_and_purpose": "e.g., Formal technical documentation; Informal customer inquiries; Persuasive marketing copy; Legal disclaimers; News reporting; etc. (semicolon delimited)"
#   }},
#   "cluster_cohesion_and_granularity": {{
#     "cohesion_level": "High | Medium | Low",
#     "justification_for_cohesion": "Briefly explain why you chose this cohesion level (e.g., 'Texts discuss the same specific feature of a product' vs. 'Texts broadly relate to finance but cover diverse subtopics').",
#     "number_of_distinct_sub_themes_identified": "Integer (e.g., 1 if highly cohesive, 2-5 if moderately diverse with clear sub-themes)"
#   }},
#   "representative_synthetic_texts_for_embedding": [ // Always provide at least one. Provide more if distinct sub-themes are identified.
#     {{
#       "rst_id": "RST_1",
#       "synthetic_text_content": "A highly specific, concise, and **semantically accurate** synthetic text for the (first) identified theme/sub-theme (english). This text must be a faithful representation ideal for precise embedding.",
#       "represented_sub_theme_description": "A brief (1 sentence) description of the specific sub-theme or nuance this RST captures, highlighting its accurate representation of the source texts.",
#       "estimated_predominant_sentiment_label": "Positive | Negative | Neutral | Mixed | Objective"
#     }}
#     // Add more RST objects here if multiple distinct sub-themes were identified and warrant separate, equally accurate RSTs.
#     // e.g., {{ "rst_id": "RST_2", "synthetic_text_content": "...", "represented_sub_theme_description": "..." }}
#   ],
#   "input_cluster_processing_recommendation": {{ // This section is about the *current input texts* you just analyzed.
#     "action_needed_before_rst_generation_is_optimal": "None | Subdivide_Input_Cluster",
#     "justification_if_subdivide_needed": "Explain ONLY if 'Subdivide_Input_Cluster' is chosen. Detail why the current input is too diverse/jumbled to generate meaningful and **accurate** RSTs, even multiple ones. (e.g., 'The texts are too disparate, covering unrelated topics like X, Y, and Z, making specific and **accurate** RST generation ineffective.')",
#     "suggested_k_for_input_sub_clustering": "If 'Subdivide_Input_Cluster', suggest an integer (e.g., 2, 3, 4) for how many sub-clusters these specific input texts should be broken into for more focused analysis and subsequent **accurate** RST generation."
#   }}
# }}
# """

#                 check_prompts[cluster_idx] = f"""
# Analyze the following text samples, which represent a cluster of texts grouped by similarity:
# {formatted_texts}

# **Important Context & Content Advisory:** The text samples you will analyze are of a political nature and are being examined as part of a social network analysis aimed at detecting potential foreign aggression. Consequently, you may encounter texts that express strong opinions, negative sentiment, aggressive language, or viewpoints that could be considered problematic. Some texts might even contain content that, in other contexts, could be flagged.

# Your objective remains to perform the analytical tasks outlined below with professional detachment. This includes:
# *   Accurately identifying themes, sentiments (including negative or aggressive ones), and key entities.
# *   Generating Representative Synthetic Texts (RSTs) that **faithfully and accurately** represent the core meaning and distinct themes of the input texts, even if those themes involve aggression or negativity. The purpose of the RSTs is to be a precise semantic representation for downstream processing, not to endorse or sanitize the original content.

# Please proceed with the analysis as instructed, focusing on accurate representation.

# Your Task:
# Your primary goal is to thoroughly analyze this cluster of texts and generate insightful outputs that can be used for both immediate understanding and for a downstream data processing pipeline. This pipeline will involve using highly accurate synthetic texts you create as new cluster centroids for the entire original corpus. The success of this downstream process hinges on the precision and faithfulness of these synthetic texts.

# Specifically, you need to:

# 1.  **Cluster Analysis:** Provide a comprehensive analysis of the given text samples. This includes identifying the central theme(s), key entities, sentiment, style, and an assessment of the cluster's internal coherence.
# 2.  **Representative Synthetic Texts (RSTs) - CRITICAL ACCURACY, FAITHFUL EMBODIMENT, AND ENGLISH OUTPUT REQUIRED:**
#     *   Generate one or more "Representative Synthetic Texts." Each RST must be a **highly accurate and semantically precise piece of text, written in ENGLISH, that faithfully *embodies and exemplifies*** a distinct theme or sub-theme found within the provided samples (even if the original samples are in a different language).
#     *   **CRUCIAL DISTINCTION: An RST is NOT a summary or description of the texts or sub-themes. It is a *new, synthetic piece of text, written in ENGLISH*, crafted *as if* it were an original sample from the cluster (translated to English if necessary), perfectly capturing the essence, style, tone, sentiment, and core message of the texts representing that specific sub-theme.**
#     *   **The content, style, and tone of each RST must be directly synthesized and derivable from the input texts, then expressed in clear, fluent ENGLISH.** Avoid introducing external concepts or creating overly broad generalizations. The RST should feel like a "perfect, distilled *example*" from the texts it represents, not an analysis of them, and rendered in English for consistent downstream processing.
#     *   **Think of it this way: If you were to translate the original texts of a sub-theme to English, and then add an RST (also in English) back into that translated set, it should blend in seamlessly, appearing as a quintessential English example of that sub-theme.**
#     *   These RSTs will be embedded using a transformer model. Texts from the original corpus will then be assigned to the RST that is semantically closest. The effectiveness of this new clustering entirely depends on the **accuracy, specificity, and representational style** of these **ENGLISH** RSTs.
#     *   If the provided text samples cover multiple discernible sub-themes, you MUST generate a separate, **equally accurate, specific, and stylistically faithful (though translated to ENGLISH)**, RST for each major sub-theme you identify. This is preferred over a single, potentially less accurate, synthetic text. Aim for 1-5 RSTs per input cluster, depending on its diversity.
# 3.  **Cohesion Assessment & Re-clustering Recommendation (for this input cluster):**
#     *   Assess the cohesion of the provided text samples.
#     *   If, and only if, the provided samples are so diverse or jumbled that you cannot generate a set of **specific, meaningful, and accurate** RSTs (even if multiple), then you should recommend that *this current input cluster sample be sub-divided further*. In this specific case, you would suggest how many sub-clusters (e.g., k for k-means) might be appropriate for these *input texts* before attempting to generate RSTs from those more coherent sub-groups. The goal is to ensure that RSTs can always be generated with high fidelity.
#     *   If you can generate good, accurate RSTs (even if multiple), then no re-clustering of the input is needed at this stage.

# Provide your response in this exact JSON format:

# {{
#   "cluster_pre_analysis_for_rst_generation": {{
#     "primary_theme_summary_brief": "Briefly state the main subject of these texts (e.g., 'Reports of drone attacks on military vehicles').",
#     "predominant_original_language": "e.g. ru, en (Report the original language)",
#     "key_recurring_elements_for_rst_crafting": {{
#       "common_actions_verbs": ["List 3-5 common actions, e.g., 'уничтожили', 'поражение', 'обнаружил', 'догнал' - list in original language"],
#       "key_technologies_equipment_mentioned": ["List 3-5, e.g., 'FPV-дрон', 'оптоволокно', 'БТР', 'танк' - list in original language"],
#       "involved_groups_actors": ["List 2-4, e.g., 'ВС РФ', 'ВСУ', 'ГВ Север', 'дроноводы' - list in original language"],
#       "typical_target_types": ["List 2-4, e.g., 'бандеромобиль', 'пехота', 'техника', 'САУ' - list in original language"]
#     }},
#     "assessment_of_sub_theme_diversity": "Based on the texts, are there distinct sub-narratives or angles? (e.g., 'Mainly one type of report' vs. 'Reports of attacks, AND reports of countermeasures, AND reports of specific drone types'). Briefly describe.",
#     "number_of_distinct_sub_themes_to_target_for_rsts": "Integer (1-5). This should match the number of RSTs you will generate. Choose based on *truly distinct* narrative angles you identified above."
#   }},
#   "representative_synthetic_texts_for_embedding": [
#     // Generate one RST object for EACH distinct sub-theme identified in 'number_of_distinct_sub_themes_to_target_for_rsts'.
#     {{
#       "rst_id": "RST_1",
#       "represented_sub_theme_description": "In 1-2 sentences, describe the SPECIFIC sub-theme, angle, or key detail THIS RST is designed to capture (e.g., 'Russian FPV drone strike on a specific type of Ukrainian vehicle, mentioning the drone's guidance system.' or 'Ukrainian attempt to counter Russian drones.' or 'Report focusing on a specific Russian military unit's drone operations.').",
#       "synthetic_text_content_in_english": "CRITICAL: This text MUST be in ENGLISH. It is NOT a summary. It is a NEW, SYNTHETIC text written AS IF it were an ORIGINAL example from this sub-theme, translated to English. It must: 1) Faithfully embody the core message, tone (including aggression/negativity if present in source), and style of the source texts for THIS sub-theme. 2) Be highly specific and incorporate relevant details (like translated versions of specific equipment, units, actions, or locations from 'key_recurring_elements_for_rst_crafting' if they fit this sub-theme). 3) Feel like a 'perfect, distilled English example' that would blend seamlessly with translated versions of the original texts for this sub-theme. AVOID generic statements. BE SPECIFIC to the sub-theme described above.",
#       "estimated_predominant_sentiment_label": "Positive | Negative | Neutral | Mixed | Objective (Reflect the sentiment of the source texts this RST represents)"
#     }}
#     // Add more RST objects here if 'number_of_distinct_sub_themes_to_target_for_rsts' > 1.
#     // e.g., {{ "rst_id": "RST_2", "represented_sub_theme_description": "...", "synthetic_text_content_in_english": "...", "estimated_predominant_sentiment_label": "..." }}
#   ],
#   "input_cluster_processing_recommendation_for_optimal_rsts": {{
#     "action_needed_if_rsts_cannot_be_specific_and_accurate": "None (RSTs above are specific and accurate for distinct sub-themes) | Subdivide_Input_Cluster_Further",
#     "justification_if_subdivide_needed": "ONLY if 'Subdivide_Input_Cluster_Further' is chosen: Explain why, even with multiple RSTs, you could not create specific, meaningful, and accurate English RSTs for distinct sub-themes from the current input. (e.g., 'Input texts are too jumbled, covering truly unrelated events X, Y, and Z, making focused RSTs impossible.')",
#     "suggested_k_for_further_sub_clustering": "If 'Subdivide_Input_Cluster_Further', suggest an integer (e.g., 2, 3) for how many sub-clusters these specific input texts might need to be broken into."
#   }}
# }}
# """


# good
                check_prompts[cluster_idx] = f"""
Analyze the following text samples, which represent a cluster of texts grouped by similarity:
{formatted_texts}

Your Task:
Your **absolute primary goal** is to generate one or more "Representative Synthetic Texts" (RSTs). These RSTs must be **exceptionally accurate, semantically precise, and directly derivable** from the provided text samples. They will serve as new cluster centroids in a downstream data processing pipeline. The success of this entire pipeline depends on the **fidelity and specificity** of the RSTs you create.

Specifically, you need to:

1.  **Internal Analysis for RST Generation:**
    *   Briefly summarize the main subject matter of the texts.
    *   Assess if the texts represent a single, highly coherent theme, or if there are multiple distinct sub-themes present. This assessment directly influences the number of RSTs to generate.
    *   Determine if the texts are coherent enough to produce accurate RSTs.

2.  **Representative Synthetic Texts (RSTs) - CRITICAL ACCURACY REQUIRED:**
    *   Generate one or more RSTs. Each RST must be a **highly accurate and semantically precise** piece of text that **faithfully epitomizes** a distinct theme or sub-theme found within the provided samples.
    *   **CRITICAL:** The content of each RST **must be directly supportable and derivable from the input texts.** Do NOT introduce external information or create generalizations that are not explicitly backed by the provided samples. Each RST should be a "perfect, distilled essence" of the texts it represents.
    *   These RSTs will be embedded, and original corpus texts will be assigned to the semantically closest RST. The effectiveness of this re-clustering entirely depends on the **accuracy and specificity** of your RSTs.
    *   If the provided texts cover multiple discernible sub-themes, you **MUST generate a separate, equally accurate and specific, RST for each major sub-theme.** Aim for 1-3 RSTs per input cluster, depending on its diversity and your ability to create truly distinct and accurate representations. If you create multiple RSTs, ensure they are genuinely different and capture unique facets of the input.

3.  **Input Cluster Viability Assessment (for this input cluster):**
    *   Based on your internal analysis, if (and only if) the provided samples are so diverse, jumbled, or contradictory that you **cannot generate a set of specific, meaningful, and accurate RSTs** (even if multiple), then recommend that this current input cluster be sub-divided further.
    *   The goal is to ensure that RSTs can *always* be generated with high fidelity. If you can generate good RSTs, no re-clustering of the input is needed.

Provide your response in this exact JSON format:

{{
  "internal_analysis_for_rst_generation": {{
    "main_subject_summary": "A concise 1-2 sentence summary of the main subject matter of the input texts.",
    "identified_sub_themes_count": "Integer (e.g., 1 if highly cohesive with a single theme; more if moderately diverse with clear, distinct sub-themes that can be accurately represented by separate RSTs).",
    "coherence_for_rst_creation": "Sufficient | Borderline | Insufficient (Sufficient if accurate RST(s) can be created. Insufficient if texts are too jumbled/diverse for accurate RSTs.)",
    "justification_for_sub_themes_and_coherence": "Briefly explain your reasoning for the sub-theme count and coherence assessment. If multiple sub-themes, briefly describe them."
  }},
  "representative_synthetic_texts_for_embedding": [ // Always provide at least one if 'coherence_for_rst_creation' is 'Sufficient' or 'Borderline'.
    {{
      "rst_id": "RST_1",
      "synthetic_text_content": "A highly specific, concise, and **semantically accurate** synthetic text (in English) for the (first) identified theme/sub-theme. This text must be a faithful representation ideal for precise embedding.",
      "represented_sub_theme_description": "A brief (1 sentence) description of the specific sub-theme or nuance this RST captures, highlighting its accurate representation of the source texts.",
      "estimated_predominant_sentiment_label": "Positive | Negative | Neutral | Mixed | Objective" // A simple sentiment label for the RST itself.
    }}
    // Add more RST objects here if 'identified_sub_themes_count' > 1 and you can create distinct, accurate RSTs.
    // e.g., {{ "rst_id": "RST_2", "synthetic_text_content": "...", "represented_sub_theme_description": "...", "estimated_predominant_sentiment_label": "..." }}
  ],
  "input_cluster_processing_recommendation": {{
    "action_needed_before_rst_generation_is_optimal": "None | Subdivide_Input_Cluster", // Choose 'Subdivide_Input_Cluster' ONLY if 'coherence_for_rst_creation' is 'Insufficient'.
    "justification_if_subdivide_needed": "Explain ONLY if 'Subdivide_Input_Cluster' is chosen. Detail why the current input is too diverse/jumbled to generate meaningful and **accurate** RSTs, even multiple ones. (e.g., 'The texts cover unrelated topics A, B, and C, making specific and **accurate** RST generation impossible at this stage.')",
    "suggested_k_for_input_sub_clustering": "If 'Subdivide_Input_Cluster', suggest an integer (e.g., 2, 3) for how many sub-clusters these specific input texts should be broken into for more focused analysis and subsequent **accurate** RST generation."
  }}
}}
"""


#test 1
#                 check_prompts[cluster_idx] = f"""
# Analyze the following text samples, which represent a cluster of texts grouped by similarity:
# {formatted_texts}

# Your Task:
# Your **absolute primary goal** is to generate one or more "Representative Synthetic Texts" (RSTs). These RSTs must be **exceptionally accurate, semantically precise, and directly derivable** from the provided text samples. They will serve as new cluster centroids in a downstream data processing pipeline. The success of this entire pipeline depends on the **fidelity and specificity** of the RSTs you create.

# Specifically, you need to:

# 1.  **Internal Analysis for RST Generation:**
#     *   Briefly summarize the main subject matter of the texts.
#     *   Assess if the texts represent a single, highly coherent theme, or if there are multiple distinct sub-themes present. This assessment directly influences the number of RSTs to generate.
#     *   Determine if the texts are coherent enough to produce accurate RSTs.

# 2.  **Representative Synthetic Texts (RSTs) - CRITICAL ACCURACY REQUIRED:**
#     *   Generate one or more RSTs. Each RST must be a **highly accurate and semantically precise** piece of text that **faithfully epitomizes** a distinct theme or sub-theme found within the provided samples.
#     *   **AVOID MERE REPHRASING OR META-DESCRIPTION:** An RST, especially if only one is generated, should not be a simple rephrasing of the `main_subject_summary`. It must be a distilled piece of *representative content* from the theme, not a description of the theme itself. If multiple RSTs are generated, each should focus on a *distinct substantive aspect* of the overall subject.
#     *   **CRITICAL:** The content of each RST **must be directly supportable and derivable from the input texts.** Do NOT introduce external information or create generalizations that are not explicitly backed by the provided samples. Each RST should be a "perfect, distilled essence" of the *informational content* of the texts it represents. **It should exemplify the *substance* of the communication, not just its form or meta-description. For example, if texts are news reports about local events, an RST should be a synthetic, concise news report about a representative event, NOT a statement like 'This cluster contains news reports about local events.'**
#     *   These RSTs will be embedded, and original corpus texts will be assigned to the semantically closest RST. The effectiveness of this re-clustering entirely depends on the **accuracy and specificity** of your RSTs.
#     *   If the provided texts cover multiple discernible sub-themes, you **MUST generate a separate, equally accurate and specific, RST for each major sub-theme.** Aim for 1-3 RSTs per input cluster, depending on its diversity and your ability to create truly distinct and accurate representations. If you create multiple RSTs, ensure they are genuinely different and capture unique facets of the input.

# 3.  **Input Cluster Viability Assessment (for this input cluster):**
#     *   Based on your internal analysis, if (and only if) the provided samples are so diverse, jumbled, or contradictory that you **cannot generate a set of specific, meaningful, and accurate RSTs** (even if multiple), then recommend that this current input cluster be sub-divided further.
#     *   The goal is to ensure that RSTs can *always* be generated with high fidelity. If you can generate good RSTs, no re-clustering of the input is needed.

# Provide your response in this exact JSON format:

# {{
#   "internal_analysis_for_rst_generation": {{
#     "main_subject_summary": "A concise 1-2 sentence summary of the main subject matter of the input texts.",
#     "identified_sub_themes_count": "Integer (e.g., 1 if highly cohesive with a single theme; 2-3 if moderately diverse with clear, distinct sub-themes that can be accurately represented by separate RSTs).",
#     "coherence_for_rst_creation": "Sufficient | Borderline | Insufficient (Sufficient if accurate RST(s) can be created. Insufficient if texts are too jumbled/diverse for accurate RSTs.)",
#     "justification_for_sub_themes_and_coherence": "Briefly explain your reasoning for the sub-theme count and coherence assessment. If multiple sub-themes, briefly describe them."
#   }},
#   "representative_synthetic_texts_for_embedding": [ // Always provide at least one if 'coherence_for_rst_creation' is 'Sufficient' or 'Borderline'.
#     {{
#       "rst_id": "RST_1",
#       "synthetic_text_content": "A highly specific, concise, and **semantically accurate** synthetic text (in English) for the (first) identified theme/sub-theme. This text must be a faithful representation ideal for precise embedding. **Crucially, the RST must synthesize the actual informational content, narrative, or viewpoint, not merely describe the types of texts, their purpose, or the categories of topics present. It should read as if it could be a new, exemplary text belonging to the theme/sub-theme itself.**",
#       "represented_sub_theme_description": "A brief (1 sentence) description of the specific sub-theme or nuance this RST captures, highlighting how it represents a distinct *type of information, core message, event, or viewpoint* found in the source texts, and confirming its accurate synthesis of that aspect from the provided samples.",
#       "estimated_predominant_sentiment_label": "Positive | Negative | Neutral | Mixed | Objective" // A simple sentiment label for the RST itself.
#     }}
#     // Add more RST objects here if 'identified_sub_themes_count' > 1 and you can create distinct, accurate RSTs.
#     // e.g., {{ "rst_id": "RST_2", "synthetic_text_content": "...", "represented_sub_theme_description": "...", "estimated_predominant_sentiment_label": "..." }}
#   ],
#   "input_cluster_processing_recommendation": {{
#     "action_needed_before_rst_generation_is_optimal": "None | Subdivide_Input_Cluster", // Choose 'Subdivide_Input_Cluster' ONLY if 'coherence_for_rst_creation' is 'Insufficient'.
#     "justification_if_subdivide_needed": "Explain ONLY if 'Subdivide_Input_Cluster' is chosen. Detail why the current input is too diverse/jumbled to generate meaningful and **accurate** RSTs, even multiple ones. (e.g., 'The texts cover unrelated topics A, B, and C, making specific and **accurate** RST generation impossible at this stage.')",
#     "suggested_k_for_input_sub_clustering": "If 'Subdivide_Input_Cluster', suggest an integer (e.g., 2, 3) for how many sub-clusters these specific input texts should be broken into for more focused analysis and subsequent **accurate** RST generation."
#   }}
# }}
# """
            
# test 2
#                 check_prompts[cluster_idx] = f"""
# Analyze the following text samples, which represent a cluster of texts grouped by similarity:
# {{formatted_texts}}

# Your Task:
# Your **absolute primary goal** is to generate one or more "Representative Synthetic Texts" (RSTs). These RSTs must be **exceptionally accurate, semantically precise, and directly derivable** from the provided text samples. They will serve as new cluster centroids in a downstream data processing pipeline. The success of this entire pipeline depends on the **fidelity and specificity** of the RSTs you create.

# Specifically, you need to:

# 1.  **Internal Analysis for RST Generation:**
#     *   Briefly summarize the main subject matter of the texts.
#     *   Assess if the texts represent a single, highly coherent theme, or if there are multiple distinct sub-themes present. This assessment directly influences the number of RSTs to generate.
#     *   Determine if the texts are coherent enough to produce accurate RSTs.

# 2.  **Representative Synthetic Texts (RSTs) - CRITICAL ACCURACY REQUIRED:**
#     *   Generate one or more RSTs. Each RST must be a **highly accurate and semantically precise** piece of text that **faithfully epitomizes** a distinct theme or sub-theme found within the provided samples.
#     *   **IMPORTANT DISTINCTION:** An RST is **NOT** merely a summary *of* the themes (that is the role of the 'main_subject_summary' in the internal analysis). Instead, an RST should read like a *prototypical example text* that could have originated from this cluster. It should be a concise, synthetic piece of content that *embodies* the theme, as if it were a new, ideal text sample from that theme.
#     *   **Think of it this way:** If the input texts were a collection of recipes for 'apple pie,' the `main_subject_summary` might be 'Recipes detailing ingredients and instructions for making apple pie.' A good RST would be a concise, synthetic recipe like: 'Combine sliced apples, cinnamon, sugar, and butter. Place in a pastry crust and bake at 375°F for 40 minutes.' A less ideal RST would be: 'This theme contains instructions for baking apple pies.'
#     *   **CRITICAL:** The content of each RST **must be directly supportable and derivable from the input texts.** Do NOT introduce external information or create generalizations that are not explicitly backed by the provided samples. Each RST should be a "perfect, distilled essence" of the texts it represents, focusing on synthesizing a *representative instance* of the content, not just describing the *category* of content.
#     *   These RSTs will be embedded, and original corpus texts will be assigned to the semantically closest RST. The effectiveness of this re-clustering entirely depends on the **accuracy and specificity** of your RSTs.
#     *   If the provided texts cover multiple discernible sub-themes, you **MUST generate a separate, equally accurate and specific, RST for each major sub-theme.** Aim for 1-3 RSTs per input cluster, depending on its diversity and your ability to create truly distinct and accurate representations. If you create multiple RSTs, ensure they are genuinely different and capture unique facets of the input.

# 3.  **Input Cluster Viability Assessment (for this input cluster):**
#     *   Based on your internal analysis, if (and only if) the provided samples are so diverse, jumbled, or contradictory that you **cannot generate a set of specific, meaningful, and accurate RSTs** (even if multiple), then recommend that this current input cluster be sub-divided further.
#     *   The goal is to ensure that RSTs can *always* be generated with high fidelity. If you can generate good RSTs, no re-clustering of the input is needed.

# Provide your response in this exact JSON format:

# {{
#   "internal_analysis_for_rst_generation": {{
#     "main_subject_summary": "A concise 1-2 sentence summary of the main subject matter of the input texts.",
#     "identified_sub_themes_count": "Integer (e.g., 1 if highly cohesive with a single theme; more if moderately diverse with clear, distinct sub-themes that can be accurately represented by separate RSTs).",
#     "coherence_for_rst_creation": "Sufficient | Borderline | Insufficient (Sufficient if accurate RST(s) can be created. Insufficient if texts are too jumbled/diverse for accurate RSTs.)",
#     "justification_for_sub_themes_and_coherence": "Briefly explain your reasoning for the sub-theme count and coherence assessment. If multiple sub-themes, briefly describe them."
#   }},
#   "representative_synthetic_texts_for_embedding": [ // Always provide at least one if 'coherence_for_rst_creation' is 'Sufficient' or 'Borderline'.
#     {{
#       "rst_id": "RST_1",
#       "synthetic_text_content": "A highly specific, concise, and **semantically accurate** synthetic text (in English) for the (first) identified theme/sub-theme. This text must be a faithful representation ideal for precise embedding, reading like a prototypical example text from the cluster.",
#       "represented_sub_theme_description": "A brief (1 sentence) description of the specific sub-theme or nuance this RST captures, highlighting its accurate representation of the source texts.",
#       "estimated_predominant_sentiment_label": "Positive | Negative | Neutral | Mixed | Objective" // A simple sentiment label for the RST itself.
#     }}
#     // Add more RST objects here if 'identified_sub_themes_count' > 1 and you can create distinct, accurate RSTs.
#     // e.g., {{ "rst_id": "RST_2", "synthetic_text_content": "...", "represented_sub_theme_description": "...", "estimated_predominant_sentiment_label": "..." }}
#   ],
#   "input_cluster_processing_recommendation": {{
#     "action_needed_before_rst_generation_is_optimal": "None | Subdivide_Input_Cluster", // Choose 'Subdivide_Input_Cluster' ONLY if 'coherence_for_rst_creation' is 'Insufficient'.
#     "justification_if_subdivide_needed": "Explain ONLY if 'Subdivide_Input_Cluster' is chosen. Detail why the current input is too diverse/jumbled to generate meaningful and **accurate** RSTs, even multiple ones. (e.g., 'The texts cover unrelated topics A, B, and C, making specific and **accurate** RST generation impossible at this stage.')",
#     "suggested_k_for_input_sub_clustering": "If 'Subdivide_Input_Cluster', suggest an integer (e.g., 2, 3) for how many sub-clusters these specific input texts should be broken into for more focused analysis and subsequent **accurate** RST generation."
#   }}
# }}
# """





#                 check_prompts[cluster_idx] = f"""
# Analyze the following text samples, which are believed to form a coherent cluster based on prior similarity analysis:
# {formatted_texts}

# Your Task:
# Your primary goal is to deeply analyze this cluster of texts. Based on your analysis, you will generate several key outputs:
# 1.  A comprehensive summary and characterization of the cluster.
# 2.  Extractable metrics that can be used for statistical analysis of a larger corpus.
# 3.  A single, synthetically generated "Representative Text". This text should encapsulate the core essence, theme, and style of the provided samples. It will be used later for embedding and assigning other texts from a larger corpus to it, effectively forming new, LLM-defined conceptual clusters.
# 4.  An assessment of the cluster's coherence for the purpose of generating this single Representative Text. If you determine that the provided samples are too diverse to be represented by a single coherent synthetic text, you will instead provide guidance on how to sub-cluster these samples.

# Detailed Instructions:

# A. Cluster Analysis & Metrics:
#     *   **Cluster Label:** Provide a concise, human-readable label (3-7 words) that best describes the overarching topic or theme of this cluster.
#     *   **Detailed Summary:** Write a 2-3 sentence summary explaining what these texts are about, their commonalities, and any discernible purpose or intent.
#     *   **Key Themes/Topics:** List key themes or sub-topics present in the texts.
#     *   **Keywords:** List 5-10 most salient keywords or keyphrases that are highly representative of this cluster.
#     *   **Dominant Sentiment (if applicable):** If a clear sentiment is expressed across the majority of texts, classify it as 'Positive', 'Negative', 'Neutral', or 'Mixed'. If not applicable or too varied, state 'Not Applicable'.
#     *   **Text Type/Genre (if discernible):** e.g., 'Customer Reviews', 'News Articles', 'Technical Support Questions', 'Social Media Posts', 'Legal Documents', 'Informal Chat'. If not clearly discernible, state 'Undetermined'.

# B. Coherence Assessment & Representative Text Generation:
#     *   **Coherence Score (1-5):** Assign a score from 1 (very diverse, multiple unrelated topics) to 5 (highly coherent, single clear topic).
#     *   **Coherence Rationale:** Briefly explain your reasoning for the coherence score.
#     *   **If Coherence Score is 3 or above (Sufficiently Coherent):**
#         *   **`generate_representative_text`:** Set to `true`.
#         *   **`representative_text_for_embedding`:** Generate a synthetic text (target 50-150 words) that acts as an idealized representative of this cluster. This text should be well-written, coherent, and capture the central theme, style, and key information points of the samples. It should be suitable for use as a "conceptual centroid" for embedding and similarity matching against a larger corpus.
#         *   **`re_clustering_needed`:** Set to `false`.
#         *   **`re_clustering_instructions`:** Set to `null`.
#     *   **If Coherence Score is 1 or 2 (Insufficiently Coherent):**
#         *   **`generate_representative_text`:** Set to `false`.
#         *   **`representative_text_for_embedding`:** Set to `null`. Do NOT attempt to generate a single representative text if the cluster is not coherent enough.
#         *   **`re_clustering_needed`:** Set to `true`.
#         *   **`re_clustering_instructions`:**
#             *   **`suggested_num_sub_clusters`:** Estimate the number of distinct sub-clusters (e.g., 2, 3) you believe exist within these samples.
#             *   **`sub_cluster_theme_descriptions`:** For each suggested sub-cluster, provide a brief (1-2 sentence) description of its likely theme. This will guide a subsequent UMAP + K-Means re-clustering process on *this specific set of texts*.

# Provide your response in this exact JSON format:

# ```json
# {{
#   "cluster_analysis": {{
#     "cluster_label": "YOUR_CONCISE_LABEL",
#     "detailed_summary": "YOUR_DETAILED_SUMMARY_OF_THE_CLUSTER",
#     "key_themes": ["Theme 1", "Theme 2", "Theme 3"],
#     "keywords": ["keyword1", "keyword2", "keyword3", "keyword4", "keyword5"],
#     "dominant_sentiment": "Positive | Negative | Neutral | Mixed | Not Applicable",
#     "text_type_genre": "Customer Reviews | News Articles | etc. | Undetermined"
#   }},
#   "coherence_and_representation": {{
#     "coherence_score": YOUR_SCORE_1_TO_5,
#     "coherence_rationale": "YOUR_REASONING_FOR_THE_SCORE",
#     "generate_representative_text": true_or_false,
#     "representative_text_for_embedding": "YOUR_SYNTHETIC_REPRESENTATIVE_TEXT_OR_NULL",
#     "re_clustering_needed": false_or_true,
#     "re_clustering_instructions": {{
#       "suggested_num_sub_clusters": null_or_INTEGER,
#       "sub_cluster_theme_descriptions": null_or_ARRAY_OF_STRINGS
#     }}
#   }}
# }}
# """

#     agenerate_text_error_addition = """# **Important Context & Content Advisory:** The text samples you will analyze are of a political nature and are being examined as part of a social network analysis aimed at detecting potential foreign aggression. Consequently, you may encounter texts that express strong opinions, negative sentiment, aggressive language, or viewpoints that could be considered problematic. Some texts might even contain content that, in other contexts, could be flagged.

# # Your objective remains to perform the analytical tasks outlined below with professional detachment. This includes:
# # *   Accurately identifying themes, sentiments (including negative or aggressive ones), and key entities.
# # *   Generating Representative Synthetic Texts (RSTs) that **faithfully and accurately** represent the core meaning and distinct themes of the input texts, even if those themes involve aggression or negativity. The purpose of the RSTs is to be a precise semantic representation for downstream processing, not to endorse or sanitize the original content."""
    
#     match_error_addition = """Your previous output has outputted an unparsable """



    tasks = []
    for cluster_idx, prompt in check_prompts.items():
        tasks.append(agenerate_text(prompt, return_token_usage=print_token_usage))
        # print(cluster_idx, prompt)
        # print(type(cluster_idx))
        # break

    results = await asyncio.gather(*tasks)
    results = {int(cluster_idx): result for cluster_idx, result in zip(check_prompts.keys(), results)}
    # print(len(results))

    coherence_and_theme = {}
    # to_redo = {"agenerate_error": [], "match_error": []}
    # to_redo = {}
    bad_cids = {}
    for k, v in results.items():
        if print_token_usage:
            v, prompt_tokens, candidate_tokens = v
            inp_tokens += prompt_tokens
            out_tokens += candidate_tokens
        # print(k, v)
        if v == "":
            print("bad cid", str(k))
            # print(k, v)
            bad_cids[str(k)] = "agenerate_text error"
            # to_redo["agenerate_error"].append({k:agenerate_text_error_addition+"\n\n"+check_prompts[k]})
            # to_redo[k] = agenerate_text_error_addition+"\n\n"+check_prompts[str(k)]

        output = []

        out_match = re.search(r"```json\s*\n(.*?)\n\s*```", v, re.DOTALL | re.IGNORECASE)

        try:
            if out_match:
                # output.append(json.loads(out_match.group(1).strip()))
                output = json.loads(out_match.group(1).strip())
        except Exception as e:
            print(k, "error", e)
            # print(k, v)
            bad_cids[str(k)] = "match error"
            # to_redo["agenerate_error"].append({k:f"Your previous output has outputted an unparsable json.\nThis was the error:\n\n{e}\n\nTry again\n\n"+check_prompts[k]})
            # to_redo[k] = "Your previous output has outputted an unparsable json.\nThis was the error:\n\n"+str(e)+"\n\nTry again\n\n"+check_prompts[str(k)]
        
        if len(output) != 0:
            coherence_and_theme[str(k)] = output

    # print("to_redo", len(to_redo))
    # tasks = []
    # for cluster_idx, prompt in to_redo.items():
    #     print("redo", cluster_idx)
    #     tasks.append(agenerate_text(prompt, return_token_usage=print_token_usage))

    # results = await asyncio.gather(*tasks)
    # results = {int(cluster_idx): result for cluster_idx, result in zip(check_prompts.keys(), results)}

    # bad_cids = {}
    # for k, v in results.items():
    #     if print_token_usage:
    #         v, prompt_tokens, candidate_tokens = v
    #         inp_tokens += prompt_tokens
    #         out_tokens += candidate_tokens

    #     if v == "":
    #         print("bad cid2", str(k))
    #         bad_cids[str(k)] = "agenerate_text error"
        
    #     out_match = re.search(r"```json\s*\n(.*?)\n\s*```", v, re.DOTALL | re.IGNORECASE)

    #     try:
    #         if out_match:
    #             output = json.loads(out_match.group(1).strip())
    #     except Exception as e:
    #         print(k, "error2", e)
    #         bad_cids[str(k)] = "match error"
        
    #     if len(output) != 0:
    #         coherence_and_theme[str(k)] = output

    
    print("tokens", inp_tokens, out_tokens)
    
    return coherence_and_theme, bad_cids