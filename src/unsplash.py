import sys
import os
from os.path import join, dirname, abspath
from dotenv import load_dotenv
import pyunsplash
import requests
from PIL import Image

# Define needed variables, including our API key

UNSPLASH_ACCESS_KEY = "soccs2fmeCHj1lY2rae7-nyKA2KfRlKN5edXOBWLC_g" 

path_film_less_filtered = '/Users/annamiraotoole/Documents/GitHub/sillystill/data/unsplash_film_less_filtered/'
path_film = '/Users/annamiraotoole/Documents/GitHub/sillystill/data/unsplash_film/'
path_digital = '/Users/annamiraotoole/Documents/GitHub/sillystill/data/unsplash_digital/'

# Initialize the Unsplash client
pu = pyunsplash.PyUnsplash(api_key=UNSPLASH_ACCESS_KEY)

# Define Functions

def save_image(path: str, old_path: str, photo: pyunsplash.src.photos.Photo):
    """
    Save the image from the given Unsplash photo object to the specified path.
    
    Args:
        path (str): The path where the image should be saved.
        photo (pyunsplash.src.photos.Photo): The Unsplash photo object.
    
    Returns:
        str: The full path and filename of the saved image.
    """
    filename = 'unsplash_' + photo.body['slug'] + '.png'
    # Check if the file already exists

    if os.path.exists(old_path + filename):
        return filename
    
    full_filename = path + filename

    response = requests.get(photo.link_download, allow_redirects=True)
    open(full_filename, 'wb').write(response.content)
    return full_filename

# filter entries based on relevancy, returns boolean
def filter_relevancy(entry: pyunsplash.src.photos.Photo) -> bool:
    """
    Filter function to determine the relevancy of a photo entry based on its description. 
    Filter for the inclusion of both 'cinestill' and '800' in the description.
    
    Args:
        entry (pyunsplash.src.photos.Photo): The photo entry to be filtered.
        
    Returns:
        bool: True if the photo entry is relevant, False otherwise.
    """
    if entry.body["description"]:
        des = entry.body["description"].lower()
        return 'cinestill' in des
    else:
        return False
    

def get_film_photos(batch: int):
    """
    Get a list of Unsplash photos based on the given page number and number of photos per page.
    
    Args:
        batch (int): Which batch of pages to get.
        
    Returns:
        Nothing
    """

    # should get 30 batches of 3-7 photos each --> 90-210 photos

    batch_size = 44

    for i in range(batch_size):

        p = batch_size * batch + i + 43 # hardcoded because we already have 43 pages of less-filtered photos saved

        print("Getting film photos page", p)
    
        # MAKE A REQUEST TO UNSPLASH API
        search_result_photos = pu.search(type_='photos', query='cinestill', page=p, per_page=10000)
        
        # filter entries based on relevancy, print out urls
        filtered_photos = filter(lambda entry: filter_relevancy(entry), search_result_photos.entries)
        # Go through filtered photos and save them to our unsplash data
        filtered_photos_lst = list(filtered_photos)
        print("After filtering, we have a total of", len(filtered_photos_lst), "relevant photos.")
        for entry in filtered_photos_lst:
            save_image(path_film_less_filtered, path_film, entry)


def get_digital_photos():
    """
    Get a list of 1000 random Unsplash photos and save them.
    """

    for i in range(5):

        print("Getting digital photos batch", i)

        # THIS CELL CAUSES A REQUEST TO UNSPLASH API
        digital_photos = pu.photos(type_='random', count=1000, featured=True)

        for entry in digital_photos.entries:
            save_image(path_digital, path_digital, entry)


if __name__ == "__main__":
    # Check if an argument is provided
    if len(sys.argv) < 2:
        print("Please provide an integer page-number argument.")
    else:
        # Get the argument from command line
        page = int(sys.argv[1])
        # Get and save the Unsplash photos (should make 45 API requests in total)
        get_film_photos(page)
        get_digital_photos()
        