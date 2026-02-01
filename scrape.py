# Usage Example:
# python -u .\scrape.py --types moth --new-species --ignore-sources insecta.pro,wikipedia.org,indianbiodiversity.org >>logs\scrape.log 2>&1

import argparse
import sys
import os
import json
import re
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from PIL import Image
import time
from datetime import datetime

# Add project root to path
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

# Ensure consistent working directory
if os.getcwd() != current_dir:
    try:
        os.chdir(current_dir)
    except Exception as e:
        print(f"Warning: Could not change directory to {current_dir}: {e}")

try:
    import mynnlib
    from mynnlib import load_json, dump_json
except ImportError:
    print("Error: mynnlib not found. Ensure you are running this from the correct directory.")
    sys.exit(1)


class BaseScraper:
    def log_header(self):
        return f"[ {threading.current_thread().name:24} ]  "

    def download_image(self, img_url, output_dir, img_timeout=30):
        try:
            img_data = requests.get(img_url, timeout=img_timeout).content
            img_name = img_url.split("/")[-1]
            # Handle potential query params in filename
            img_name = img_name.split('?')[0]
            
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
                
            img_path = os.path.join(output_dir, img_name)
            with open(img_path, 'wb') as file:
                file.write(img_data)
            return True
        except Exception as e:
            # print(f"{self.log_header()}{e}")
            return False

    def has_parent_with_prop(self, tag, prop, value, max_parents):
        parent = tag
        for i in range(0, max_parents):
            parent = parent.parent
            if parent is None:
                return False
            if parent.get(prop) == value:
                return True
        return False

class MothsOfIndiaScraper(BaseScraper):
    def __init__(self):
        self.dataset_dir = "insect-dataset/src/mothsofindia.org"
        self.website_url = "https://www.mothsofindia.org"
        self.initial_path = "/lepidoptera"
        self.page_timeout = 120
        self.image_timeout = 30
        self.ignore_image_regex = r"^(imgs10|.*(boimobileapp|butterfliesofurbangreeneries|webheader|headerlogo|WPA-[IVX]+).*)\.(png|jpg|jpeg)$"
        self.early_stage_suffix = '-early'
        self.new_species_url = f"{self.website_url}/history-of-moth-species-pages-new"
        
    def scrape_images(self, url, output_dir, skip_existing_dir=False):
        if skip_existing_dir and os.path.exists(output_dir) and os.listdir(output_dir):
             return True

        try:
            print(f"{self.log_header()}    Scraping URL: {url}")
            response = requests.get(url, timeout=self.page_timeout)
            soup = BeautifulSoup(response.text, 'html.parser')
            img_tags = soup.find_all('img')
            success_cnt = 0
            failure_cnt = 0
            already_downloaded_cnt = 0
            for img in img_tags:
                img_url = img.get('src')
                if self.has_parent_with_prop(img, 'id', 'laraval', 5):
                    continue
                class_suffix = ''
                if self.has_parent_with_prop(img, 'id', 'early', 6):
                    class_suffix = self.early_stage_suffix
                if img_url:
                    img_url = urljoin(url, img_url)
                    img_name = img_url.split("/")[-1]
                    if re.search(self.ignore_image_regex, img_name):
                        continue
                    
                    target_dir = output_dir + class_suffix
                    if os.path.exists(os.path.join(target_dir, img_name)):
                        already_downloaded_cnt += 1
                        continue
                    
                    if self.download_image(img_url, target_dir, self.image_timeout):
                        success_cnt += 1
                    else:
                        failure_cnt += 1
            if success_cnt > 0:
                print(f"{self.log_header()}      Downloaded {success_cnt}(+{already_downloaded_cnt}) / {success_cnt+already_downloaded_cnt+failure_cnt} image(s) in {output_dir}")
            return True
        except Exception as e:
            print(f"{self.log_header()}{e}")
            return False

    def fetch_missing_species(self):
        print(f"{self.log_header()}Fetching new species list from {self.new_species_url}")
        try:
            response = requests.get(self.new_species_url, timeout=self.page_timeout)
            soup = BeautifulSoup(response.text, 'html.parser')
            content_div = soup.find(class_="node__content")
            links = content_div.find_all('a') if content_div else []
            
            missing_species = set()
            existing_species = set()
            if os.path.exists(self.dataset_dir):
                existing_species = {name.lower() for name in os.listdir(self.dataset_dir)}
                
            ignore_regex = r"(spp|genera|spp\.|group)$"
            
            for link in links:
                href = link.get('href')
                if not href:
                    continue
                
                if href.startswith(self.website_url):
                    href = href.replace(self.website_url, "")
                    
                if href.startswith("#") or "javascript:" in href:
                    continue
                
                if "?" in href:
                    href = href.split("?")[0]
                    
                species_name = href.strip("/")
                
                if not species_name:
                    continue

                if not species_name[0].isupper():
                    continue
                    
                normalized_name = species_name.lower()
                
                if '/' in normalized_name:
                    continue
                    
                if re.search(ignore_regex, normalized_name):
                    continue
                    
                if normalized_name not in existing_species:
                    missing_species.add(normalized_name)
                    
            result = sorted(list(missing_species))
            print(f"{self.log_header()}Found {len(result)} missing species")
            return result

        except Exception as e:
            print(f"{self.log_header()}Error fetching missing species: {e}")
            return []

class IndianOdonataScraper(BaseScraper):
    def __init__(self):
        self.dataset_dir = "insect-dataset/src/indianodonata.org"
        self.website_url = "https://www.indianodonata.org"
        self.page_timeout = 120
        self.image_timeout = 30
        self.ignore_image_regex = r"^(imgs10|.*(iucn-red-list|mobileapp|butterfliesofurbangreeneries|webheader|headerlogo|WPA-[IVX]+).*)\.(png|jpg|jpeg)$"
        self.early_stage_suffix = '-early'

    def scrape_images(self, url, output_dir, skip_existing_dir=False):
        if skip_existing_dir and os.path.exists(output_dir) and os.listdir(output_dir):
             return True

        try:
            print(f"{self.log_header()}    Scraping URL: {url}")
            response = requests.get(url, timeout=self.page_timeout)
            soup = BeautifulSoup(response.text, 'html.parser')
            img_tags = soup.find_all('img')
            success_cnt = 0
            failure_cnt = 0
            already_downloaded_cnt = 0
            for img in img_tags:
                img_url = img.get('src')
                if self.has_parent_with_prop(img, 'id', 'laraval', 5):
                    continue
                class_suffix = ''
                if self.has_parent_with_prop(img, 'id', 'early', 6):
                    class_suffix = self.early_stage_suffix
                if img_url:
                    img_url = urljoin(url, img_url)
                    img_name = img_url.split("/")[-1]
                    if re.search(self.ignore_image_regex, img_name):
                        continue
                    
                    target_dir = output_dir + class_suffix
                    if os.path.exists(os.path.join(target_dir, img_name)):
                        already_downloaded_cnt += 1
                        continue
                    
                    if self.download_image(img_url, target_dir, self.image_timeout):
                        success_cnt += 1
                    else:
                        failure_cnt += 1
            if success_cnt > 0:
                print(f"{self.log_header()}      Downloaded {success_cnt}(+{already_downloaded_cnt}) / {success_cnt+already_downloaded_cnt+failure_cnt} image(s) in {output_dir}")
            return True
        except Exception as e:
            print(f"{self.log_header()}{e}")
            return False

    def fetch_missing_species(self):
        # Explicitly unimplemented as requested
        return []

class IFoundButterfliesScraper(BaseScraper):
    def __init__(self):
        self.dataset_dir = "insect-dataset/src/ifoundbutterflies.org"
        self.website_url = "https://www.ifoundbutterflies.org"
        self.page_timeout = 120
        self.image_timeout = 30
        self.ignore_image_regex = r"^(imgs10|.*(boimobileapp|butterfliesofurbangreeneries|webheader|headerlogo|WPA-[IVX]+).*)\.(png|jpg|jpeg)$"
        self.early_stage_suffix = '-early'

    def scrape_images(self, url, output_dir, skip_existing_dir=False):
        if skip_existing_dir and os.path.exists(output_dir) and os.listdir(output_dir):
             return True

        try:
            print(f"{self.log_header()}    Scraping URL: {url}")
            response = requests.get(url, timeout=self.page_timeout)
            soup = BeautifulSoup(response.text, 'html.parser')
            img_tags = soup.find_all('img')
            success_cnt = 0
            failure_cnt = 0
            already_downloaded_cnt = 0
            for img in img_tags:
                img_url = img.get('src')
                if self.has_parent_with_prop(img, 'id', 'laraval', 5):
                    continue
                class_suffix = ''
                if self.has_parent_with_prop(img, 'id', 'early', 6):
                    class_suffix = self.early_stage_suffix
                if img_url:
                    img_url = urljoin(url, img_url)
                    img_name = img_url.split("/")[-1].split("?")[0]
                    if re.search(self.ignore_image_regex, img_name):
                        continue
                    
                    target_dir = output_dir + class_suffix
                    if os.path.exists(os.path.join(target_dir, img_name)):
                        already_downloaded_cnt += 1
                        continue
                    
                    if self.download_image(img_url, target_dir, self.image_timeout):
                        success_cnt += 1
                    else:
                        failure_cnt += 1
            if success_cnt > 0:
                print(f"{self.log_header()}      Downloaded {success_cnt}(+{already_downloaded_cnt}) / {success_cnt+already_downloaded_cnt+failure_cnt} image(s) in {output_dir}")
            return True
        except Exception as e:
            print(f"{self.log_header()}{e}")
            return False

class IndianCicadasScraper(BaseScraper):
    def __init__(self):
        self.dataset_dir = "insect-dataset/src/indiancicadas.org"
        self.website_url = "https://www.indiancicadas.org"
        self.page_timeout = 120
        self.image_timeout = 30
        self.ignore_image_regex = r"^(imgs10|.*(iucn-red-list|mobileapp|butterfliesofurbangreeneries|webheader|headerlogo|WPA-[IVX]+).*)\.(png|jpg|jpeg)$"
        self.early_stage_suffix = '-early'

    def scrape_images(self, url, output_dir, skip_existing_dir=False):
        if skip_existing_dir and os.path.exists(output_dir) and os.listdir(output_dir):
             return True

        try:
            print(f"{self.log_header()}    Scraping URL: {url}")
            response = requests.get(url, timeout=self.page_timeout)
            soup = BeautifulSoup(response.text, 'html.parser')
            img_tags = soup.find_all('img')
            success_cnt = 0
            failure_cnt = 0
            already_downloaded_cnt = 0
            for img in img_tags:
                img_url = img.get('src')
                if self.has_parent_with_prop(img, 'id', 'laraval', 5):
                     continue
                class_suffix = ''
                if self.has_parent_with_prop(img, 'id', 'early', 6):
                     class_suffix = self.early_stage_suffix
                if img_url:
                    img_url = urljoin(url, img_url)
                    img_name = img_url.split("/")[-1].split("?")[0]
                    if re.search(self.ignore_image_regex, img_name):
                        continue
                    
                    target_dir = output_dir + class_suffix
                    if os.path.exists(os.path.join(target_dir, img_name)):
                        already_downloaded_cnt += 1
                        continue
                    
                    if self.download_image(img_url, target_dir, self.image_timeout):
                        success_cnt += 1
                    else:
                        failure_cnt += 1
            if success_cnt > 0:
                print(f"{self.log_header()}      Downloaded {success_cnt}(+{already_downloaded_cnt}) / {success_cnt+already_downloaded_cnt+failure_cnt} image(s) in {output_dir}")
            return True
        except Exception as e:
            print(f"{self.log_header()}{e}")
            return False

class INaturalistScraper(BaseScraper):
    def __init__(self):
        self.dataset_dir = "insect-dataset/src/inaturalist.org"
        self.page_timeout = 120
        self.image_timeout = 30
        self.max_workers = 50

    def check_image(self, file_path):
        try:
            with Image.open(file_path) as img:
                img.verify()
            return True
        except (IOError, SyntaxError):
            return False
        
    def download_image_inat(self, img_url, output_dir, uuid):
        try:
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
            img_name = img_url.split("/")[-1]
            img_path = os.path.join(output_dir, f"{uuid}-{img_name.split('?')[0]}")
            if Path(img_path).is_file() and self.check_image(img_path):
                return 'EXISTS'
            img_data = requests.get(img_url, timeout=self.image_timeout).content
            with open(img_path, 'wb') as file:
                file.write(img_data)
            if not self.check_image(img_path):
                print(f"{self.log_header()}Removing corrupted image {file.name}")
                os.remove(Path(img_path))
                if not os.listdir(output_dir):
                    os.rmdir(output_dir)
                return 'FAILURE'
            return 'SUCCESS'
        except Exception as e:
            if os.path.exists(output_dir) and not os.listdir(output_dir):
                os.rmdir(output_dir)
            return 'FAILURE'

    def get_observations(self, taxon_id, page):
        url = "https://api.inaturalist.org/v1/observations"
        params = {
            "taxon_id": taxon_id,
            "order_by": "votes",
            "quality_grade": "research",
            "photos": "true",
            "page": page,
            "per_page": 100
        }
        headers = {
            "Accept": "application/json",
        }
        return requests.get(url, params=params, headers=headers)

    def find_taxon_id(self, soup):
        tag_a = soup.find("a", class_="name sciname")
        if not tag_a:
            parent_div = soup.find("div", class_="taxonimage")
            if not parent_div:
                parent_div = soup.find("div", class_="first")
            if parent_div:
                tag_a = parent_div.find("a")
        if tag_a:
            return re.sub(r"(/taxa/)|(-.+$)", "", tag_a.get("href"))
        return None

    def scrape(self, class_names, skip_existing_dir=False):
        success_cnt = 0
        failure_cnt = 0
        exists_cnt = 0
        for class_name in class_names:
            if skip_existing_dir and os.path.exists(f"{self.dataset_dir}/{class_name}"):
                continue
            try:
                url = f"https://www.inaturalist.org/taxa/search?q={class_name}"
                response = requests.get(url, timeout=self.page_timeout)
                soup = BeautifulSoup(response.text, 'html.parser')
                taxon_id = self.find_taxon_id(soup)
                if not taxon_id:
                    continue
                response = self.get_observations(taxon_id, 1)
                for result in response.json().get("results", []):
                    suffixed_class_name = class_name
                    if "tags" in result and len([ t for t in result["tags"] if re.match(r"^.*(egg|eggs|larva|larvae|pupa|pupae).*$", t)]) > 0:
                        suffixed_class_name += "-early"
                    for observation in result.get("observation_photos", []):
                        if "photo" in observation:
                            img_url = re.sub(r"\bsquare\b", "medium", observation["photo"]["url"])
                            status = self.download_image_inat(img_url, f"{self.dataset_dir}/{suffixed_class_name}", observation["uuid"])
                            success_cnt += 1 if status=='SUCCESS' else 0
                            failure_cnt += 1 if status=='FAILURE' else 0
                            exists_cnt += 1 if status=='EXISTS' else 0
                if success_cnt + failure_cnt + exists_cnt > 0:
                    print(f"{self.log_header()}Processed {class_name} | taxon_id:{taxon_id}")
                    print(f"{self.log_header()}SUCCESS: {success_cnt:5} | FAILURE: {failure_cnt:5} | EXISTS: {exists_cnt:5}")
            except Exception as ex:
                continue
    
    def scrape_multithread(self, class_names, batch_size, skip_existing_dir=False):
        print(f"{self.log_header()}Starting iNaturalist scraping...")
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = [executor.submit(self.scrape, class_names[offset:(offset+batch_size)], skip_existing_dir) for offset in range(0, len(class_names), batch_size)]
            for future in futures:
                result = future.result()


def get_species_list(species_json, family, group_name):
    if family in species_json:
        for group in species_json[family]['species']:
            if group.get('group') == group_name:
                return group['names']
    return []

def main():
    print(f"Script started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    parser = argparse.ArgumentParser(description="Insect Data Scraper")
    # Removed --sources, added --ignore-sources
    parser.add_argument("--types", nargs="+", default=["moth", "odonata"], 
                        help="List of insect types (e.g., moth, butterfly, odonata, cicada)")
    parser.add_argument("--ignore-sources", type=str, default="", 
                        help="Comma-separated list of sources to ignore (e.g., insecta.pro,wikipedia.org)")
    parser.add_argument("--new-species", action="store_true", 
                        help="Skip species for which a directory already exists")
    
    args = parser.parse_args()
    ignored_sources = [s.strip() for s in args.ignore_sources.split(",") if s.strip()]

    # Define Source Map
    SOURCE_MAP = {
        "moth": ['inaturalist.org', 'mothsofindia.org', 'insecta.pro', 'wikipedia.org', 'indianbiodiversity.org'],
        "butterfly": ['ifoundbutterflies.org', 'inaturalist.org', 'insecta.pro', 'wikipedia.org', 'indianbiodiversity.org'],
        "odonata": ['indianodonata.org', 'inaturalist.org', 'insecta.pro', 'wikipedia.org', 'indianbiodiversity.org'],
        "cicada": ['indiancicadas.org', 'inaturalist.org', 'insecta.pro', 'wikipedia.org', 'indianbiodiversity.org']
    }

    # Load species.json
    species_json = load_json("species.json")
    
    # Initialize Scrapers
    # Only initializing implemented scrapers for now
    scrapers = {
        "mothsofindia.org": MothsOfIndiaScraper(),
        "indianodonata.org": IndianOdonataScraper(),
        "inaturalist.org": INaturalistScraper(),
        "ifoundbutterflies.org": IFoundButterfliesScraper(),
        "indiancicadas.org": IndianCicadasScraper(),
    }

    for insect_type in args.types:
        print(f"Processing type: {insect_type}")
        
        # Resolve species list dynamically
        current_species_list = []
        family = ""
        group_name = ""
        
        if insect_type == "moth":
            family = 'lepidoptera'
            group_name = 'Moths'
        elif insect_type == "butterfly":
             family = 'lepidoptera'
             group_name = 'Butterflies'
        elif insect_type == "odonata":
             family = 'odonata'
             group_name = 'All'
        elif insect_type == "cicada":
             family = 'hemiptera'
             group_name = 'Cicada'
        
        if family and group_name:
             current_species_list = get_species_list(species_json, family, group_name)

        if not current_species_list:
            print(f"Warning: No species list found for type '{insect_type}' (Family: {family}, Group: {group_name}). Skipping.")
            continue

        # Get intended sources for this type
        sources = SOURCE_MAP.get(insect_type, [])
        
        # 1. Fetch new species (Only for moth currently supported/requested)
        if insect_type == "moth" and "mothsofindia.org" in sources and "mothsofindia.org" not in ignored_sources:
             scraper = scrapers.get("mothsofindia.org")
             if scraper:
                found = scraper.fetch_missing_species()
                if found:
                    initial_set = set(current_species_list)
                    found_set = set(found)
                    really_new = found_set - initial_set
                    
                    if really_new:
                        updated_list = sorted(list(initial_set.union(found_set)))
                        
                        # Update JSON safely
                        if 'lepidoptera' in species_json:
                            for group in species_json['lepidoptera']['species']:
                                if group.get('group') == 'Moths':
                                    group['names'] = updated_list
                                    dump_json("species.json", species_json)
                                    current_species_list = updated_list
                                    print(f"Added {len(really_new)} new species to species.json: {list(really_new)}")
                                    break
                    else:
                        print(f"No NEW species added to species.json (Found {len(found)} missing from disk, but they were already in JSON).")
        
        # 2. Iterate through sources
        for source in sources:
            if source in ignored_sources:
                continue

            scraper = scrapers.get(source)
            if not scraper:
                # print(f"Scraper for {source} is not implemented yet. Skipping.")
                continue

            print(f"Scraping {source} for {insect_type}...")
            
            if source == "inaturalist.org":
                scraper.scrape_multithread(current_species_list, batch_size=1, skip_existing_dir=args.new_species)
            else:
                 for species in current_species_list:
                    if args.new_species and os.path.exists(f"{scraper.dataset_dir}/{species}"):
                         continue
                    species_url = f"{scraper.website_url}/{species}"
                    scraper.scrape_images(species_url, f"{scraper.dataset_dir}/{species}", skip_existing_dir=args.new_species)

if __name__ == "__main__":
    main()
