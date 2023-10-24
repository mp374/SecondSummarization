import glob
import json
import os

def read_file_content(file_path):

    list_of_story_bodies = []
    list_of_story_summaries = []
    # Open the json file.
    story_file = open(file_path)
    # convert the json file content into a dictionary.
    file_content = json.load(story_file)

    # Iterating through the json
    # list
    for story in file_content:
        list_of_story_bodies.append(story["body"])
        list_of_story_summaries.append(story["title"])
    # Closing file
    story_file.close()

    return list_of_story_bodies, list_of_story_summaries


def get_list_all_stories():

    path = '/Users/heshankavinda/Library/CloudStorage/OneDrive-UniversityofPlymouth/PROJ518/data set/Sample'
    all_stories = []
    all_summaries = []
    for filename in glob.glob(os.path.join(path, '*.json')):
        stories, summaries = read_file_content(filename)
        all_stories.extend(stories)
        all_summaries.extend(summaries)

    print(len(glob.glob(os.path.join(path, '*.json'))), "files opened and ", len(all_stories), "stories read.")
    return all_stories, all_summaries