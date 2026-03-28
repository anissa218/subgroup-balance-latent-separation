import torch
import pandas as pd
import numpy as np
import os

from src.utils.civilcomments_attr_definitions import GROUP_ATTRS, AGGREGATE_ATTRS, ORIG_ATTRS

### GENERAL ###

def make_proportion_dfs(df, col, total_size, seed = 42):
    '''
    Make dfs with specific proportions of a certain subgroup
    '''
    group0 = df[df[col] == 0]
    group1 = df[df[col] == 1]
    results = []

    for pct in range(0, 110, 10):  # from 0 to 100 (inclusive) step 10
        prop_1 = pct / 100
        n1 = int(prop_1 * total_size)
        n0 = total_size - n1

        sampled_0 = group0.sample(n=n0, replace=False, random_state=seed)
        sampled_1 = group1.sample(n=n1, replace=False, random_state=seed)
        combined = pd.concat([sampled_0, sampled_1]).sample(frac=1, random_state=seed).reset_index(drop=True)
        results.append((pct, combined))

    return results
    
### MNIST ###
def add_colour_fg(img_tensor,colour_channel):
    '''
    img_tensor: torch.tensor with values between 0 and 255 of dim [28,28]
    colour_channel: int, colour channel to add (can be multiple), either 0 or 1
    returns: 3 channel image tensor [3,H,W]
    '''
    rgb_imgs = torch.zeros(3,img_tensor.shape[0],img_tensor.shape[1],dtype=img_tensor.dtype)

    mask = img_tensor > 0  # Only modify where img_tensor has non-zero values
    rgb_imgs[colour_channel, :, :][mask] = img_tensor[mask]
    rgb_imgs[(colour_channel + 1) % 3, :, :][mask] = (img_tensor[mask] * torch.rand(mask.sum())).to(img_tensor.dtype) # randomly add a bit of other colours (not to background)
    rgb_imgs[(colour_channel + 2) % 3, :, :][mask] = (img_tensor[mask] * torch.rand(mask.sum())).to(img_tensor.dtype)

    return rgb_imgs
    
def make_subset_images(images_df,images_tensor,fg_colour_channel_col='Sex_binary',bg_colour_channel_col = 'Artefact',noise=1):
    '''
    images_df: df with col 'image_index' and 'image_colour_channel'
    images_tensor: [n_images,28,28] tensor
    return tensor of all images with colour: [n_images,3,28,28]
    '''
    subset_images = torch.zeros(len(images_df),3,images_tensor.shape[1],images_tensor.shape[2])
    for i in range(len(images_df)):
        index = images_df.iloc[i]['image_index']
        fg_colour_channel = images_df.iloc[i][fg_colour_channel_col]
        rgb_image = add_colour_fg(images_tensor[index],fg_colour_channel)
        subset_images[index] = rgb_image
    return subset_images

### Civil_comments ###

def load_df(root):
    """
    Loads the data and removes all examples where we don't have identity annotations.
    """
    df = pd.read_csv(os.path.join(root, 'all_data.csv'))
    df = df.loc[(df['identity_annotator_count'] > 0), :]
    df = df.reset_index(drop=True)
    return df

def augment_df(df):
    """
    Augment the dataframe with auxiliary attributes.
    First, we create aggregate attributes, like `LGBTQ` or `other_religions`.
    These are aggregated because there would otherwise not be enough examples to accurately
    estimate their accuracy.

    Next, for each category of demographics (e.g., race, gender), we construct an auxiliary
    attribute (e.g., `na_race`, `na_gender`) that is 1 if the comment has no identities related to
    that demographic, and is 0 otherwise.
    Note that we can't just create a single multi-valued attribute like `gender` because there's
    substantial overlap: for example, 4.6% of comments mention both male and female identities.
    """
    df = df.copy()
    for aggregate_attr in AGGREGATE_ATTRS:
        aggregate_mask = pd.Series([False] * len(df))
        for attr in AGGREGATE_ATTRS[aggregate_attr]:
            attr_mask = (df[attr] >= 0.5)
            aggregate_mask = aggregate_mask | attr_mask
        df[aggregate_attr] = 0
        df.loc[aggregate_mask, aggregate_attr] = 1

    attr_count = np.zeros(len(df))
    for attr in ORIG_ATTRS:
        attr_mask = (df[attr] >= 0.5)
        attr_count += attr_mask
    df['num_identities'] = attr_count
    df['more_than_one_identity'] = (attr_count > 1)

    for group in GROUP_ATTRS:
        print(f'## {group}')
        counts = {}
        na_mask = np.ones(len(df))
        for attr in GROUP_ATTRS[group]:
            attr_mask = (df[attr] >= 0.5)
            na_mask = na_mask & ~attr_mask
            counts[attr] = np.mean(attr_mask)
        counts['n/a'] = np.mean(na_mask)

        col_name = f'{group}_any'
        df[col_name] = 1
        df.loc[na_mask, col_name] = 0 # swaped these around!

        for k, v in counts.items():
            print(f'{k:40s}: {v:.4f}')
        print()
    return df