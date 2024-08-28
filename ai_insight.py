import json
import requests
import secrets
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
import numpy as np
import re

class SiteInformation:
    def __init__(self, name, url_home, url_username_format, username_claimed,
                information, is_nsfw, username_unclaimed=secrets.token_urlsafe(10)):
        """Create Site Information Object.

        Contains information about a specific website.

        Keyword Arguments:
        self                   -- This object.
        name                   -- String which identifies site.
        url_home               -- String containing URL for home of site.
        url_username_format    -- String containing URL for Username format
                                  on site.
                                  NOTE:  The string should contain the
                                         token "{}" where the username should
                                         be substituted.  For example, a string
                                         of "https://somesite.com/users/{}"
                                         indicates that the individual
                                         usernames would show up under the
                                         "https://somesite.com/users/" area of
                                         the website.
        username_claimed       -- String containing username which is known
                                  to be claimed on website.
        username_unclaimed     -- String containing username which is known
                                  to be unclaimed on website.
        information            -- Dictionary containing all known information
                                  about website.
                                  NOTE:  Custom information about how to
                                         actually detect the existence of the
                                         username will be included in this
                                         dictionary.  This information will
                                         be needed by the detection method,
                                         but it is only recorded in this
                                         object for future use.
        is_nsfw                -- Boolean indicating if site is Not Safe For Work.

        Return Value:
        Nothing.
        """

        self.name = name
        self.url_home = url_home
        self.url_username_format = url_username_format

        self.username_claimed = username_claimed
        self.username_unclaimed = username_unclaimed
        self.information = information
        self.is_nsfw  = is_nsfw

    def __str__(self):
        """Convert Object To String.

        Keyword Arguments:
        self                   -- This object.

        Return Value:
        Nicely formatted string to get information about this object.
        """
        
        return f"{self.name} ({self.url_home})"


class SitesInformation:
    def __init__(self, data_file_path=None):
        """Create Sites Information Object.

        Contains information about all supported websites.

        Keyword Arguments:
        self                   -- This object.
        data_file_path         -- String which indicates path to data file.
                                  The file name must end in ".json".

                                  There are 3 possible formats:
                                   * Absolute File Format
                                     For example, "c:/stuff/data.json".
                                   * Relative File Format
                                     The current working directory is used
                                     as the context.
                                     For example, "data.json".
                                   * URL Format
                                     For example,
                                     "https://example.com/data.json", or
                                     "http://example.com/data.json".

                                  An exception will be thrown if the path
                                  to the data file is not in the expected
                                  format, or if there was any problem loading
                                  the file.

                                  If this option is not specified, then a
                                  default site list will be used.

        Return Value:
        Nothing.
        """

        if not data_file_path:
            data_file_path = "https://raw.githubusercontent.com/sherlock-project/sherlock/master/sherlock_project/resources/data.json"

        if not data_file_path.lower().endswith(".json"):
            raise FileNotFoundError(f"Incorrect JSON file extension for data file '{data_file_path}'.")

        if data_file_path.lower().startswith("http"):
            try:
                response = requests.get(url=data_file_path)
                response.raise_for_status()
                site_data = response.json()
            except requests.RequestException as error:
                raise FileNotFoundError(
                    f"Problem while attempting to access data file URL '{data_file_path}': {error}"
                )
            except ValueError as error:
                raise ValueError(
                    f"Problem parsing json contents at '{data_file_path}': {error}."
                )
        else:
            try:
                with open(data_file_path, "r", encoding="utf-8") as file:
                    site_data = json.load(file)
            except FileNotFoundError:
                raise FileNotFoundError(f"Problem while attempting to access data file '{data_file_path}'.")
            except ValueError as error:
                raise ValueError(
                    f"Problem parsing json contents at '{data_file_path}': {error}."
                )

        site_data.pop('$schema', None)

        self.sites = {}
        self._initialize_sites(site_data)

    def _initialize_sites(self, site_data):
        """Initialize sites from data.

        Keyword Arguments:
        self                   -- This object.
        site_data               -- Dictionary containing site data.

        Return Value:
        None
        """
        for site_name in site_data:
            try:
                self.sites[site_name] = SiteInformation(
                    site_name,
                    site_data[site_name]["urlMain"],
                    site_data[site_name]["url"],
                    site_data[site_name]["username_claimed"],
                    site_data[site_name],
                    site_data[site_name].get("isNSFW", False)
                )
            except KeyError as error:
                raise ValueError(
                    f"Problem parsing json contents: Missing attribute {error}."
                )
            except TypeError:
                print(f"Encountered TypeError parsing json contents for target '{site_name}'\nSkipping target.\n")

    def remove_nsfw_sites(self, do_not_remove: list = []):
        """
        Remove NSFW sites from the sites, if isNSFW flag is true for site

        Keyword Arguments:
        self                   -- This object.
        do_not_remove          -- List of site names that should not be removed, case-insensitive.

        Return Value:
        None
        """
        sites = {}
        do_not_remove = [site.casefold() for site in do_not_remove]
        for site in self.sites:
            if self.sites[site].is_nsfw and site.casefold() not in do_not_remove:
                continue
            sites[site] = self.sites[site]  
        self.sites = sites

    def site_name_list(self):
        """Get Site Name List.

        Keyword Arguments:
        self                   -- This object.

        Return Value:
        List of strings containing names of sites.
        """

        return sorted([site.name for site in self], key=str.lower)

    def analyze_site_descriptions(self):
        """Analyze site descriptions using NLP.

        Keyword Arguments:
        self                   -- This object.

        Return Value:
        None
        """
        descriptions = [site.information.get('description', '') for site in self.sites.values()]
        vectorizer = TfidfVectorizer(stop_words='english')
        X = vectorizer.fit_transform(descriptions)
        kmeans = KMeans(n_clusters=5, random_state=0).fit(X)
        clusters = kmeans.predict(X)

        for i, site in enumerate(self.sites.values()):
            site.information['cluster'] = int(clusters[i])
        print("Site descriptions analyzed and clustered.")

    def detect_anomalies(self):
        """Detect anomalies in site data.

        Keyword Arguments:
        self                   -- This object.

        Return Value:
        None
        """
        patterns = {
            'url': r'^https?://\S+\.\S+$',
            'username_claimed': r'^\w+$'
        }

        for site in self.sites.values():
            for key, pattern in patterns.items():
                if key in site.information and not re.match(pattern, site.information[key]):
                    print(f"Anomaly detected for site '{site.name}': Invalid '{key}'")

    def __iter__(self):
        """Iterator For Object.

        Keyword Arguments:
        self                   -- This object.

        Return Value:
        Iterator for sites object.
        """

        for site_name in self.sites:
            yield self.sites[site_name]

    def __len__(self):
        """Length For Object.

        Keyword Arguments:
        self                   -- This object.

        Return Value:
        Length of sites object.
        """
        return len(self.sites)
