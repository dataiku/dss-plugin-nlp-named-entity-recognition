# -*- coding: utf-8 -*-
import logging
from time import time

import dataiku
from dataiku.runnables import Runnable

from ner_utils_flair import CustomSequenceTagger


class MyRunnable(Runnable):
    def __init__(self, project_key, config, plugin_config):
        """
        :param project_key: the project in which the runnable executes
        :param config: the dict of the configuration of the object
        :param plugin_config: contains the plugin settings
        """
        self.project_key = project_key
        self.config = config
        self.plugin_config = plugin_config
        self.client = dataiku.api_client()

    def get_progress_target(self):
        """
        If the runnable will return some progress info, have this function return a tuple of
        (target, unit) where unit is one of: SIZE, FILES, RECORDS, NONE
        """
        return (100, "NONE")

    def run(self, progress_callback):

        # Retrieving parameters
        output_folder_name = self.config.get("folder_name", "")

        # Creating new Managed Folder if needed
        project = self.client.get_project(self.project_key)
        output_folder_found = False
        for folder in project.list_managed_folders():
            if output_folder_name == folder["name"]:
                output_folder = project.get_managed_folder(folder["id"])
                output_folder_found = True
                break
        if not output_folder_found:
            output_folder = project.create_managed_folder(output_folder_name)
        output_folder = dataiku.Folder(output_folder.get_definition()["id"], project_key=self.project_key)
        if output_folder.get_info().get("type") != "Filesystem":
            raise TypeError("Please store the model on the server filesystem")
        output_folder_path = output_folder.get_path()

        logging.info("Downloading Flair model...")
        start = time()
        CustomSequenceTagger.load(model="ner-ontonotes-fast", cache_path=output_folder_path)
        result_message = "Downloading Flair model: Done in {:.2f} seconds.".format(time() - start)
        logging.info(result_message)
        return result_message
