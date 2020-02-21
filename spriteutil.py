#!/usr/bin/env python3
import timeit
import numpy as np
import sys
import random
from PIL import Image, ImageDraw


# WP1
def find_most_common_color(image):
    """
    Returns the pixel color that is the most used in an image.

    @params:
    1. image: an image object.

    @return:
    1. an integer if the mode is grayscale;
    2. a tuple (red, green, blue) of integers (0 to 255) if the mode is RGB;
    3. a tuple (red, green, blue, alpha) of integers (0 to 255) if the mode is RGBA.
    """
    try:
        colors_list = image.getcolors(image.width * image.height) # unsorted list of (count, pixel) values
        return max(colors_list, key=lambda tup:tup[0])[1]

    except AttributeError:
        raise TypeError("Not an Image object!")


# WP2
class Sprite:
    """
    Class Sprite which constructor takes 5 arguments label, x1, y1, x2, and y2.
    """

    def __init__(self, label, x1, y1, x2, y2):
        """
        Initialize class Sprite's attributes.

        @params:
        label, x1, y1, x2, y2: positive integers.

        @return:
        1. label: the label of the sprite.
        2. top_left: (x1, y1) (a tuple) of the top-left corner.
        3. bottom_right: (x2, y2) (a tuple) of the right-most corner.
        4. width: the number of pixels horizontally of the sprite.
        5. height: the number of pixels vertically of the sprite.
        """

        try:
            if label >=0 and \
                x1 >=0 and \
                y1 >=0 and \
                x2 >=0 and \
                y2 >=0 and \
                x1 <=x2 and \
                y1 <=y2:
                self.__label = label
                self.__top_left = (x1, y1)
                self.__bottom_right = (x2, y2)
                self.__width = x2 - x1 + 1
                self.__height = y2 - y1 + 1
            else:
                raise ValueError("Invalid coordinates")
        except TypeError:
            raise ValueError("Invalid coordinates")

    @property
    def label(self):
        """
        Add the read-only property 'label' that returns the label of the sprite.
        """
        return self.__label
    
    @property
    def top_left(self):
        """
        Add the read-only property 'top_left' that returns the coordinates of the top-left corner of the sprite.
        """
        return self.__top_left
    
    @property
    def bottom_right(self):
        """
        Add the read-only property 'bottom_right' that returns the coordinates of the right-most corner of the sprite.
        """
        return self.__bottom_right
    
    @property
    def width(self):
        """
        Add the read-only property 'width' that returns the number of pixels horizontally of the sprite.
        """
        return self.__width
    
    @property
    def height(self):
        """
        Add the read-only property 'height' that returns the number of pixels vertically of the sprite.
        """
        return self.__height


def check_neighborhood(pixel_coordinate, labelled_dict):
    """
    Check nearest neighbors and return a list of equivalence for that pixel.
    @params:
    1. pixel_coordinate: coordinate of the pixel.
    2. labelled_dict: dict containing label of each image pixels {label: [pixels]}.

    @return:
    1. a set of pixel's equivalence.
    """
    # check neighborhood to find exist label
    equivalence = set()
    directions = [(-1,-1), (0,-1), (1,-1), (-1,0), (1,0), (-1,1), (0,1), (1,1)]
    for direction in directions:
        neighbor = tuple(map(lambda x,y: x+y, pixel_coordinate, direction))
        for label,list_pixels in labelled_dict.items():

            # if neighbor has a label
            if neighbor in list_pixels:
                equivalence.add(label)
    return equivalence


def find_label(pixel_coordinate, labelled_dict, equivalence_list):
    """
    Return a label for the pixel and a list contain all pixels' equivalence (CCL Pass 1).
    @params:
    1. pixel coordinate: coordinate of the pixel (x,y).
    2. labelled_dict: dict containing label of each image pixels {label: [pixels]}.
    3. equivalence_list: list containing sets of all pixels' equivalence [set(equivalence)].

    @return:
    1. label: temporary label for the pixel after first pass.
    2. equivalence_list: list containing sets of all pixels' equivalence [set(equivalence)].
    """
    if not labelled_dict:
        label = 1
    else:

        # check label based on neighbors
        equivalence = check_neighborhood(pixel_coordinate, labelled_dict)
        if equivalence:
            # print(equivalence)
            label = min(equivalence)
            equivalence_list.append(equivalence)
            
        # if all neighbors have no label.
        else:

            # create new label when neighborhood have no label
            label = max(labelled_dict.keys()) + 1
    return label, equivalence_list


def update_label(labelled_dict, equivalence_list):
    """
    Return a dictionary containing label for each pixels {label: [pixels]} (CCL Pass 2).
    """
    equivalence_dict = {}
    for label in labelled_dict:
        for sets in equivalence_list:
            if label in sets:
                if label in equivalence_dict:
                    equivalence_dict[label].update(sets)
                else:
                    equivalence_dict[label] = sets
    for key, value in equivalence_dict.items():
        equivalence_dict[key] = min(value)
    # print(equivalence_dict)
    
    for label in labelled_dict.copy():
        reduced_label = equivalence_dict[label]
        if reduced_label != label:
            labelled_dict[reduced_label] += labelled_dict[label]
            del labelled_dict[label]
    return labelled_dict


def create_labelled_dict(image, background_color):
    """
    """
    labelled_dict = {}
    equivalence_list = []

    # loop though image pixels (label pass 1)
    for y in range(image.height):
        for x in range(image.width):
            pixel_color = image.getpixel((x, y))

            # if pixel belong to a sprite
            if pixel_color != background_color:

                # get pixel coordinate
                pixel_coordinate = (x,y)

                # get pixel label
                pixel_label, equivalence_list = find_label(pixel_coordinate, labelled_dict, equivalence_list)

                # append coordinate to labelled_dict if pixel's label existed
                if pixel_label in labelled_dict:
                    labelled_dict[pixel_label].append(pixel_coordinate)
                
                # create new key in labelled_dict for new label
                else:
                    labelled_dict[pixel_label] = [pixel_coordinate]
    
    # update label based on connected component label (pass 2)
    labelled_dict = update_label(labelled_dict, equivalence_list)
    return labelled_dict


def create_label_map(labelled_dict, image):
    """
    Return A 2D array of integers of equal dimension (width and height) as the original
    image where the sprites are packed in. The label_map array maps each pixel of the image
    passed to the function to the label of the sprite this pixel corresponds to, or 0 if this
    pixel doesn't belong to a sprite (e.g., background color).

    @params:

    @return:
    label_map: 2D array of integers containing sprites as image. 
    """
    label_map = []
    for y in range(image.height):
        horizontal_label_map = []
        for x in range(image.width):
            pixel_coordinate = (x,y)
            pixel_label = 0
            for label,list_pixels in labelled_dict.items():
                if pixel_coordinate in list_pixels:
                    pixel_label = label
            horizontal_label_map.append(pixel_label)
        label_map.append(horizontal_label_map)
    label_map = np.array(label_map)
    return label_map


# WP3
def find_sprites(image, background_color=None):
    """
    """
    if background_color is None:
        background_color = find_most_common_color(image)
    labelled_dict = create_labelled_dict(image, background_color)

    # print total sprites
    print(len(labelled_dict))

    # list of sprites
    sprites = {}
    for label in labelled_dict:
        x1 = min(labelled_dict[label], key=lambda x: x[0])[0]
        x2 = max(labelled_dict[label], key=lambda x: x[0])[0]
        y1 = min(labelled_dict[label], key=lambda x: x[1])[1]
        y2 = max(labelled_dict[label], key=lambda x: x[1])[1]
        sprite = Sprite(label, x1, y1, x2, y2)
        sprites[label] = sprite
    
    # label_map
    label_map = create_label_map(labelled_dict, image)

    return sprites, label_map


# WP4
def create_sprite_labels_image(sprites, label_map, background_color=(255, 255, 255)):
    """
    """
    width = len(label_map[0])
    height = len(label_map)
    a = 255
    if len(background_color) == 3:
        background_color += (a,)
    label_color = {}
    empty_arr = np.zeros((height,width,4), dtype=np.uint8)
    for (x,y), label in np.ndenumerate(label_map):
        if label == 0:
            empty_arr[x][y] = background_color
        else:
            if label not in label_color:
                r,g,b = random.sample(range(0, 255), 3)
                random_color = (r,g,b,a)
                label_color[label] = random_color
            empty_arr[x][y] = label_color[label]
    img = Image.fromarray(empty_arr)

    draw = ImageDraw.Draw(img)
    for label, sprite in sprites.items():
        top_left = sprite.top_left
        bottom_right = sprite.bottom_right
        top_right = tuple(map(lambda x,y: x+y, top_left, (sprite.width-1, 0)))
        bottom_left =  tuple(map(lambda x,y: x+y, top_left, (0, sprite.height-1)))
        draw.line([top_left,top_right], fill=label_color[label])
        draw.line([top_left,bottom_left], fill=label_color[label])
        draw.line([bottom_right,top_right], fill=label_color[label])
        draw.line([bottom_right,bottom_left], fill=label_color[label])
    return img


def main():
    """
    Main function
    """
    # Test WP1
    # image = Image.open("./2d_video_game_commando.jpg") # rgb
    # image = Image.open("./metal_slug_sprite_standing_stance_large.png") # rgba
    # image = Image.open("./2d_video_game_metal_slug.png") # rgba
    # image = Image.open("./2d_video_game_guerilla_war.png") # P
    # image = image.convert("L") # L
    # print(image.mode)
    # background_color = find_most_common_color(image)
    # print(background_color)
    # print(timeit.timeit(stmt=lambda: find_most_common_color(image), number=1)) # measure the execution time

    # Test WP2
    # sprite = Sprite(1, 12, 23, 145, 208)
    # sprite = Sprite(1, -1, 0, 0, 0)
    # sprite = Sprite(1, "Hello", 0, 0, 0)
    # sprite = Sprite(1, 1, 0, 0, 0)
    # print(sprite.label)
    # print(sprite.top_left)
    # print(sprite.bottom_right)
    # print(sprite.width)
    # print(sprite.height)

    # Test WP3
    # image = Image.open("./metal_slug_single_sprite.png")
    image = Image.open("./optimized_sprite_sheet.png")
    sprites, label_map = find_sprites(image)
    for label, sprite in sprites.items():
        print(f"Sprite ({label}): [{sprite.top_left}, {sprite.bottom_right}] {sprite.width}x{sprite.height}")
    np.set_printoptions(threshold=sys.maxsize)
    # print(label_map)
    # print(timeit.timeit(stmt=lambda: find_sprites(image), number=1)) # measure the execution time

    # Test WP4
    sprite_label_image = create_sprite_labels_image(sprites, label_map)
    sprite_label_image.save('optimized_sprite_sheet_bounding_box_white_background.png')
    print(timeit.timeit(stmt=lambda: create_sprite_labels_image(sprites, label_map), number=1)) # measure the execution time

    sprite_label_image = create_sprite_labels_image(sprites, label_map, background_color=(0, 0, 0, 0))
    sprite_label_image.save('optimized_sprite_sheet_bounding_box_transparent_background.png')
    print(timeit.timeit(stmt=lambda: create_sprite_labels_image(sprites, label_map, background_color=(0, 0, 0, 0)), number=1)) # measure the execution time


if __name__ == "__main__":
    main()