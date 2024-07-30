# Format: model_name-rgba-cutout_name

# SAM_v1 : weight=50, threshold=25, decay=0.99
# SAM_v2 : weight=150, threshold=25, decay=0.9
# aclip_v1 : weight=25, threshold=25, decay=0.99
# aclip_v2 : weight=150, threshold=25, decay=0.9  # default in the code
# aclip_v1_matte : weight=25, threshold=25, decay=0.99, mask_proposal=vit-matte
# aclip_v2_matte : weight=150, threshold=25, decay=0.9, mask_proposal=vit-matte

settings = {
    'pas-md-euler-neg-exc': {'grad_weight': 0},  # MD
    'pas-euler-neg-exc': {'do_md': 'False', 'grad_weight': 0},  # Nothing
    'pas-suffix-euler-neg-exc': {'do_md': 'False', 'use_suffix': 'True', 'grad_weight': 0},  # Suffix
    'pas-md-sam_v2-euler-neg-exc': {'gradient_model': 'sam'},   # MD + SAM_v2
    'pas-sam_v2-euler-neg-exc': {'do_md': 'False', 'gradient_model': 'sam'},  # Nothing + SAM_v2
    'pas-suffix-sam_v2-euler-neg-exc': {'do_md': 'False', 'use_suffix': 'True', 'gradient_model': 'sam'},  # Suffix + SAM_v2
    'pas-suffix-aclip_v2-euler-neg-exc': {'do_md': 'False', 'use_suffix': 'True'},  # Suffix + aclip_v2
    'pas-suffix-aclip_v2_matte-euler-neg-exc': {'do_md': 'False', 'use_suffix': 'True', 'mask_proposal': 'vit-matte'},  # Suffix + aclip_v2_matte
    'pas-aclip_v2-euler-neg-exc': {'do_md': 'False'},  # Nothing + aclip_v2
    'pas-aclip_v2_matte-euler-neg-exc': {'do_md': 'False', 'mask_proposal': 'vit-matte'},  # Nothing + aclip_v2_matte
    'pas-md-a_clip_v2-euler-neg-exc': {},  # MD + aclip_v2
    'pas-md-a_clip_v2_matte-euler-neg-exc': {'mask_proposal': 'vit-matte'},  # MD + aclip_v2_matte
    'pas-md-sam_v1-euler-neg-exc': {  # MD + SAM_v1
        'gradient_model': 'sam', 'grad_weight': 50, 'grad_decay_rate': 0.99},
    'pas-sam_v1-euler-neg-exc': {  # Nothing + SAM_v1
        'do_md': 'False', 'gradient_model': 'sam', 'grad_weight': 50, 'grad_decay_rate': 0.99},
    'pas-suffix-sam_v1-euler-neg-exc': {  # Suffix + SAM_v1
        'do_md': 'False', 'use_suffix': 'True', 'gradient_model': 'sam', 'grad_weight': 50, 'grad_decay_rate': 0.99},
    'pas-suffix-aclip_v1-euler-neg-exc': {   # Suffix + aclip_v1
        'do_md': 'False', 'use_suffix': 'True', 'grad_weight': 25, 'grad_decay_rate': 0.99},
    'pas-aclip_v1-euler-neg-exc': {'do_md': 'False', 'grad_weight': 25, 'grad_decay_rate': 0.99},  # Nothing + aclip_v1
    'pas-md-a_clip_v1-euler-neg-exc': {'grad_weight': 25, 'grad_decay_rate': 0.99,},  # MD + aclip_v1
    'pas-suffix-aclip_v1_matte-euler-neg-exc': {   # Suffix + aclip_v1_matte
        'do_md': 'False', 'use_suffix': 'True', 'grad_weight': 25, 'grad_decay_rate': 0.99, 'mask_proposal': 'vit-matte'},
    'pas-aclip_v1_matte-euler-neg-exc': {'do_md': 'False', 'grad_weight': 25, 'grad_decay_rate': 0.99, 'mask_proposal': 'vit-matte'},  # Nothing + aclip_v1_matte
    'pas-md-a_clip_v1_matte-euler-neg-exc': {'grad_weight': 25, 'grad_decay_rate': 0.99, 'mask_proposal': 'vit-matte'},  # MD + aclip_v1_matte


    # Grabcut cutout
    'pas-md-euler-neg-rgba-gb_v1': {'grad_weight': 0, 'exclude_generic_nouns': 'False'},  # MD
    'pas-md-euler-neg-exc-rgba-gb_v1': {'grad_weight': 0, 'exclude_generic_nouns': 'False'},  # MD NO GENERIC
    'pas-euler-neg-exc-rgba-gb_v1': {'do_md': 'False', 'grad_weight': 0},  # Nothing
    'pas-suffix-euler-neg-exc-rgba-gb_v1': {'do_md': 'False', 'use_suffix': 'True', 'grad_weight': 0},  # Suffix
    'pas-md-sam_v2-euler-neg-exc-rgba-gb_v1': {'gradient_model': 'sam'},   # MD + SAM_v2
    'pas-sam_v2-euler-neg-exc-rgba-gb_v1': {'do_md': 'False', 'gradient_model': 'sam'},  # Nothing + SAM_v2
    'pas-suffix-sam_v2-euler-neg-exc-rgba-gb_v1': {'do_md': 'False', 'use_suffix': 'True', 'gradient_model': 'sam'},  # Suffix + SAM_v2
    'pas-suffix-aclip_v2-euler-neg-exc-rgba-gb_v1': {'do_md': 'False', 'use_suffix': 'True'},  # Suffix + aclip_v2
    'pas-aclip_v2-euler-neg-exc-rgba-gb_v1': {'do_md': 'False'},  # Nothing + aclip_v2
    'pas-md-a_clip_v2-euler-neg-exc-rgba-gb_v1': {},  # MD + aclip_v2
    'pas-md-sam_v1-euler-neg-exc-rgba-gb_v1': {  # MD + SAM_v1
        'gradient_model': 'sam', 'grad_weight': 50, 'grad_decay_rate': 0.99},
    'pas-sam_v1-euler-neg-exc-rgba-gb_v1': {  # Nothing + SAM_v1
        'do_md': 'False', 'gradient_model': 'sam', 'grad_weight': 50, 'grad_decay_rate': 0.99},
    'pas-suffix-sam_v1-euler-neg-exc-rgba-gb_v1': {  # Suffix + SAM_v1
        'do_md': 'False', 'use_suffix': 'True', 'gradient_model': 'sam', 'grad_weight': 50, 'grad_decay_rate': 0.99},
    'pas-suffix-aclip_v1-euler-neg-exc-rgba-gb_v1': {   # Suffix + aclip_v1
        'do_md': 'False', 'use_suffix': 'True', 'grad_weight': 25, 'grad_decay_rate': 0.99},
    'pas-aclip_v1-euler-neg-exc-rgba-gb_v1': {'do_md': 'False', 'grad_weight': 25, 'grad_decay_rate': 0.99},  # Nothing + aclip_v1
    'pas-md-a_clip_v1-euler-neg-exc-rgba-gb_v1': {'grad_weight': 25, 'grad_decay_rate': 0.99},  # MD + aclip_v1
    'pas-suffix-aclip_v2_matte-euler-neg-exc-rgba-gb_v1': {'do_md': 'False', 'use_suffix': 'True', 'mask_proposal': 'vit-matte'},  # Suffix + aclip_v2_matte
    'pas-aclip_v2_matte-euler-neg-exc-rgba-gb_v1': {'do_md': 'False', 'mask_proposal': 'vit-matte'},  # Nothing + aclip_v2_matte
    'pas-md-a_clip_v2_matte-euler-neg-exc-rgba-gb_v1': {'mask_proposal': 'vit-matte'},  # MD + aclip_v2_matte
    'pas-suffix-aclip_v1_matte-euler-neg-exc-rgba-gb_v1': {   # Suffix + aclip_v1_matte
        'do_md': 'False', 'use_suffix': 'True', 'grad_weight': 25, 'grad_decay_rate': 0.99, 'mask_proposal': 'vit-matte'},
    'pas-aclip_v1_matte-euler-neg-exc-rgba-gb_v1': {'do_md': 'False', 'grad_weight': 25, 'grad_decay_rate': 0.99, 'mask_proposal': 'vit-matte'},  # Nothing + aclip_v1_matte
    'pas-md-a_clip_v1_matte-euler-neg-exc-rgba-gb_v1': {'grad_weight': 25, 'grad_decay_rate': 0.99, 'mask_proposal': 'vit-matte'},  # MD + aclip_v1_matte


    # vit-matte cutout
    'pas-md-euler-neg-exc-rgba-vit_matte': {'grad_weight': 0, 'cutout_model': 'vit-matte'},  # MD
    'pas-md-euler-neg-rgba-vit_matte': {'grad_weight': 0, 'cutout_model': 'vit-matte', 'exclude_generic_nouns': 'False'},  # MD WITH GENERIC NOUNS
    'pas-euler-neg-exc-rgba-vit_matte': {'do_md': 'False', 'grad_weight': 0, 'cutout_model': 'vit-matte'},  # Nothing
    'pas-suffix-euler-neg-exc-rgba-vit_matte': {'do_md': 'False', 'use_suffix': 'True', 'grad_weight': 0, 'cutout_model': 'vit-matte'},  # Suffix
    'pas-md-sam_v2-euler-neg-exc-rgba-vit_matte': {'gradient_model': 'sam', 'cutout_model': 'vit-matte'},   # MD + SAM_v2
    'pas-sam_v2-euler-neg-exc-rgba-vit_matte': {'do_md': 'False', 'gradient_model': 'sam', 'cutout_model': 'vit-matte'},  # Nothing + SAM_v2
    'pas-suffix-sam_v2-euler-neg-exc-rgba-vit_matte': {'do_md': 'False', 'use_suffix': 'True', 'gradient_model': 'sam', 'cutout_model': 'vit-matte'},  # Suffix + SAM_v2
    'pas-suffix-aclip_v2-euler-neg-exc-rgba-vit_matte': {'do_md': 'False', 'use_suffix': 'True', 'cutout_model': 'vit-matte'},  # Suffix + aclip_v2
    'pas-aclip_v2-euler-neg-exc-rgba-vit_matte': {'do_md': 'False', 'cutout_model': 'vit-matte'},  # Nothing + aclip_v2
    'pas-md-a_clip_v2-euler-neg-exc-rgba-vit_matte': {'cutout_model': 'vit-matte'},  # MD + aclip_v2
    'pas-md-sam_v1-euler-neg-exc-rgba-vit_matte': {  # MD + SAM_v1
        'gradient_model': 'sam', 'grad_weight': 50, 'grad_decay_rate': 0.99, 'cutout_model': 'vit-matte'},
    'pas-sam_v1-euler-neg-exc-rgba-vit_matte': {  # Nothing + SAM_v1
        'do_md': 'False', 'gradient_model': 'sam', 'grad_weight': 50, 'grad_decay_rate': 0.99, 'cutout_model': 'vit-matte'},
    'pas-suffix-sam_v1-euler-neg-exc-rgba-vit_matte': {  # Suffix + SAM_v1
        'do_md': 'False', 'use_suffix': 'True', 'gradient_model': 'sam', 'grad_weight': 50, 'grad_decay_rate': 0.99, 'cutout_model': 'vit-matte'},
    'pas-suffix-aclip_v1-euler-neg-exc-rgba-vit_matte': {   # Suffix + aclip_v1
        'do_md': 'False', 'use_suffix': 'True', 'grad_weight': 25, 'grad_decay_rate': 0.99, 'cutout_model': 'vit-matte'},
    'pas-aclip_v1-euler-neg-exc-rgba-vit_matte': {'do_md': 'False', 'grad_weight': 25, 'grad_decay_rate': 0.99, 'cutout_model': 'vit-matte'},  # Nothing + aclip_v1
    'pas-md-a_clip_v1-euler-neg-exc-rgba-vit_matte': {'grad_weight': 25, 'grad_decay_rate': 0.99, 'cutout_model': 'vit-matte'},  # MD + aclip_v1
    'pas-suffix-aclip_v2_matte-euler-neg-exc-rgba-vit_matte': {'do_md': 'False', 'use_suffix': 'True', 'mask_proposal': 'vit-matte', 'cutout_model': 'vit-matte'},  # Suffix + aclip_v2_matte
    'pas-aclip_v2_matte-euler-neg-exc-rgba-vit_matte': {'do_md': 'False', 'mask_proposal': 'vit-matte', 'cutout_model': 'vit-matte'},  # Nothing + aclip_v2_matte
    'pas-md-a_clip_v2_matte-euler-neg-exc-rgba-vit_matte': {'mask_proposal': 'vit-matte', 'cutout_model': 'vit-matte'},  # MD + aclip_v2_matte
    'pas-suffix-aclip_v1_matte-euler-neg-exc-rgba-vit_matte': {   # Suffix + aclip_v1_matte
        'do_md': 'False', 'use_suffix': 'True', 'grad_weight': 25, 'grad_decay_rate': 0.99, 'mask_proposal': 'vit-matte', 'cutout_model': 'vit-matte'},
    'pas-aclip_v1_matte-euler-neg-exc-rgba-vit_matte': {'do_md': 'False', 'grad_weight': 25, 'grad_decay_rate': 0.99, 'mask_proposal': 'vit-matte', 'cutout_model': 'vit-matte'},  # Nothing + aclip_v1_matte
    'pas-md-a_clip_v1_matte-euler-neg-exc-rgba-vit_matte': {'grad_weight': 25, 'grad_decay_rate': 0.99, 'mask_proposal': 'vit-matte', 'cutout_model': 'vit-matte'},  # MD + aclip_v1_matte




}