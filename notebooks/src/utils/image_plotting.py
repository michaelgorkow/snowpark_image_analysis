import pandas as pd
import matplotlib.pyplot as plt
from skimage import io
from skimage.transform import resize
from textwrap import wrap
import numpy as np
import snowflake.snowpark.functions as F
from pandarallel import pandarallel
import seaborn as sns
from PIL import Image, ImageFont, ImageDraw

pandarallel.initialize(progress_bar=False)

def show_images_from_df(df, relative_path='RELATIVE_PATH', stage='@', title='RELATIVE_PATH', ncol=2, max_images=10, resize_shape=None, figsize=None):
    # remove timestamp columns
    df = df.drop([col[0] for col in df.dtypes if col[1] == 'timestamp'])
    df = df.sample(n=max_images).cache_result()
    # Download all unique images in parallel
    unique_images = df[[relative_path]].distinct()
    unique_images = unique_images.to_df(['RELATIVE_PATH'])
    unique_images = unique_images.with_column('PRESIGNED_URL', F.call_builtin('GET_PRESIGNED_URL', stage, F.col('RELATIVE_PATH'))).to_pandas()
    unique_images['IMAGE'] = unique_images.parallel_apply(lambda x: io.imread(x['PRESIGNED_URL']), axis=1)

    # resize
    if resize_shape is not None:
        unique_images['IMAGE'] = unique_images.parallel_apply(lambda x: resize(x['IMAGE'], resize_shape), axis=1)

    # Download DataFrame
    df = df.to_pandas()

    # Build plot
    nimages = len(df)
    if nimages > ncol:
        nrow = nimages // ncol + 1
    else:
        nrow = 1
        ncol = nimages
    if figsize is None:
        figsize = (16, 16 // ncol * nrow)
    fig = plt.figure(figsize=figsize)
    for i in range(nimages):
        image = unique_images[unique_images.RELATIVE_PATH == df[relative_path][i]]['IMAGE'].values[0]
        if resize_shape is not None:
            image = resize(image, resize_shape)
        if title is not None:
            label = df[title][i]
        else:
            label = 'N/A'
        ax = fig.add_subplot(nrow, ncol, i + 1)
        ax.set_title('{}'.format(label))
        plt.imshow(image)
        plt.xticks([]), plt.yticks([])
    plt.show()

def show_similar_images(df, relative_path_left='RELATIVE_PATH_LEFT', relative_path_right='RELATIVE_PATH_RIGHT', stage='@', max_similar_images=2, distance_col = None, resize_shape=None, figsize=None):
    # Download all unique images in parallel
    unique_images = df[[relative_path_left]].union_all(df[[relative_path_right]]).distinct()
    unique_images = unique_images.to_df(['RELATIVE_PATH'])
    unique_images = unique_images.with_column('PRESIGNED_URL', F.call_builtin('GET_PRESIGNED_URL', stage, F.col('RELATIVE_PATH'))).to_pandas()
    unique_images['IMAGE'] = unique_images.parallel_apply(lambda x: io.imread(x['PRESIGNED_URL']), axis=1)

    # resize
    if resize_shape is not None:
        unique_images['IMAGE'] = unique_images.parallel_apply(lambda x: resize(x['IMAGE'], resize_shape), axis=1)

    # Download DataFrame
    df = df.to_pandas()

    # Build plot
    fig = plt.figure(figsize=figsize)
    num_unique_images = len(unique_images)
    plot_row = 0
    for left_image in df[relative_path_left].unique():
        right_images = df[df[relative_path_left] == left_image].reset_index()
        ax = fig.add_subplot(num_unique_images, max_similar_images+1, (plot_row*(max_similar_images+1))+1)
        ax.set_title('{}'.format(left_image))
        plt.imshow(unique_images[unique_images.RELATIVE_PATH == left_image]['IMAGE'].values[0])
        for ix, row in right_images.iterrows():
            ax = fig.add_subplot(num_unique_images, max_similar_images+1, (plot_row*(max_similar_images+1))+ix+2)
            if distance_col is not None:
                ax.set_title('{} - Distance: {}'.format(row[relative_path_right],row[distance_col]))
            else:
                ax.set_title('{}'.format(row[relative_path_right]))
            plt.imshow(unique_images[unique_images.RELATIVE_PATH == row[relative_path_right]]['IMAGE'].values[0])
        plot_row += 1
    plt.show()

def show_classifications_from_df(df, relative_path='RELATIVE_PATH', stage='@', title='RELATIVE_PATH', classifications='CLASSIFICATIONS', max_images=10, resize_shape=None, figsize=(15,20)):
    # remove timestamp columns
    df = df.drop([col[0] for col in df.dtypes if col[1] == 'timestamp'])
    df = df.sample(n=max_images).cache_result()
    # Download all unique images in parallel
    unique_images = df[[relative_path]].distinct()
    unique_images = unique_images.to_df(['RELATIVE_PATH'])
    unique_images = unique_images.with_column('PRESIGNED_URL', F.call_builtin('GET_PRESIGNED_URL', stage, F.col('RELATIVE_PATH'))).to_pandas()
    unique_images['IMAGE'] = unique_images.parallel_apply(lambda x: io.imread(x['PRESIGNED_URL']), axis=1)

    # resize
    if resize_shape is not None:
        unique_images['IMAGE'] = unique_images.parallel_apply(lambda x: resize(x['IMAGE'], resize_shape), axis=1)

    # Download DataFrame
    df = df.to_pandas()

    # Build the plot
    row_ix = 1
    fig = plt.figure(figsize=figsize)
    for ix, row in df.iterrows():
        # Image Plot
        image = unique_images[unique_images.RELATIVE_PATH == row[relative_path]]['IMAGE'].values[0]
        ax = fig.add_subplot(len(df), 2, ix+row_ix)
        ax.set_title('{}'.format(row[title]))
        plt.xticks([]), plt.yticks([])
        plt.imshow(image)
        # Probability Plot
        ax2 = fig.add_subplot(len(df), 2, ix+row_ix+1)
        labels = pd.DataFrame(eval(row[classifications]))['label'].tolist()
        labels = [ '\n'.join(wrap(l, 40)) for l in labels ]
        scores = pd.DataFrame(eval(row[classifications]))['score'].tolist()
        y_pos = (0.2 + np.arange(len(labels))) / (1 + len(labels))
        width = 0.8 / (1 + len(labels))
        colors = ['blue', 'green', 'yellow', 'orange', 'red']
        for i in range(len(labels)):
            ax2.barh(y_pos[i], scores[i], width, align='center',
                    color=colors[i], ecolor='black')
            ax2.text(scores[i] + 0.01, y_pos[i], '{:.2%}'.format(scores[i]))
        ax2.set_yticks(y_pos)
        ax2.set_yticklabels(labels)
        ax2.set_xlabel('Probability')
        ax2.set_xticks([0, 0.25, 0.5, 0.75, 1])
        ax2.set_xticklabels(['0%', '25%', '50%', '75%', '100%'])
        ax2.set_title('Predicted Probability')
        fig.subplots_adjust(left=None, bottom=None, right=1.5, top=None, wspace=None, hspace=0.5)
        row_ix += 1


def show_object_detections_from_df(df, relative_path='RELATIVE_PATH', stage='@', title='RELATIVE_PATH', max_images=10, resize_shape=None, figsize=None):
    # remove timestamp columns
    df = df.drop([col[0] for col in df.dtypes if col[1] == 'timestamp'])
    filter_list = list(df[['RELATIVE_PATH']].distinct().limit(max_images).to_pandas().values.flatten())
    df = df[df['RELATIVE_PATH'].isin(filter_list)].cache_result()
    # Download all unique images in parallel
    unique_images = df[[relative_path]].distinct()
    unique_images = unique_images.to_df(['RELATIVE_PATH'])
    unique_images = unique_images.with_column('PRESIGNED_URL', F.call_builtin('GET_PRESIGNED_URL', stage, F.col('RELATIVE_PATH'))).to_pandas()
    unique_images['IMAGE'] = unique_images.parallel_apply(lambda x: io.imread(x['PRESIGNED_URL']), axis=1)
    unique_images['IMAGE_WIDTH'] = unique_images.parallel_apply(lambda x: x['IMAGE'].shape[1], axis=1)
    unique_images['IMAGE_HEIGHT'] = unique_images.parallel_apply(lambda x: x['IMAGE'].shape[0], axis=1)

    # get unique objects for color mapping
    unique_objects = df[['OBJECT_LABEL']].distinct().to_pandas().values
    # create a color palette
    color_palette = sns.color_palette("husl", len(unique_objects))
    color_palette = [(int(color[0]*255), int(color[1]*255), int(color[2]*255)) for color in color_palette]
    object_color_mapping = {k[0]:v for k,v in zip(unique_objects,color_palette)}

    # resize
    if resize_shape is not None:
        unique_images['IMAGE'] = unique_images.parallel_apply(lambda x: (resize(x['IMAGE'], resize_shape)*255).astype(np.uint8), axis=1)

    # Download DataFrame
    df = df.to_pandas()

    # Build plot
    fig = plt.figure(figsize=figsize)
    font = ImageFont.truetype('../resources/fonts//arial.ttf',16)
    image_ix = 1
    for ix, row in unique_images.iterrows():
        image = Image.fromarray(row['IMAGE'])
        if resize_shape is not None:
            image_width_scaling = resize_shape[1] / row['IMAGE_WIDTH']
            image_height_scaling = resize_shape[0] / row['IMAGE_HEIGHT']
        else:
            image_width_scaling = 1
            image_height_scaling = 1
        draw = ImageDraw.Draw(image)
        objects_df = df[df[relative_path] == row['RELATIVE_PATH']]
        # draw bounding boxes
        for ix2, row2 in objects_df.iterrows():
            draw.rectangle(((int(row2['OBJECT_XMIN']*image_width_scaling),int(row2['OBJECT_YMIN']*image_height_scaling)), 
                            (int(row2['OBJECT_XMAX']*image_width_scaling), int(row2['OBJECT_YMAX'])*image_height_scaling)), width=5, outline=object_color_mapping[row2['OBJECT_LABEL']])
            draw.rectangle(((row2['OBJECT_XMIN']*image_width_scaling,row2['OBJECT_YMIN']*image_height_scaling-20), 
                            (row2['OBJECT_XMIN']*image_width_scaling+int(font.getlength(row2['OBJECT_LABEL'])), row2['OBJECT_YMIN']*image_height_scaling)), width=5, fill=object_color_mapping[row2['OBJECT_LABEL']])
            draw.text((row2['OBJECT_XMIN']*image_width_scaling,row2['OBJECT_YMIN']*image_height_scaling-20),row2['OBJECT_LABEL'],(255,255,255), font=font)
        ax = fig.add_subplot(len(unique_images), 1, image_ix)
        ax.set_title('{}'.format(row[title]))
        plt.xticks([]), plt.yticks([])
        plt.imshow(image)
        image_ix += 1
    fig.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=None, hspace=0.1)
    plt.show()
    return