
import sys
import os

# Add parent directory to path using absolute path to persist across chdir
sys.path.append(os.path.abspath(".."))
os.chdir("D:/Projects/insect-id")

import mynnlib
from mynnlib import *
import re
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin
import threading
from concurrent.futures import ThreadPoolExecutor


class MothsOfIndiaScraper:
    def __init__(self):
        self.dataset_dir = "insect-dataset/src/mothsofindia.org"
        self.website_url = "https://www.mothsofindia.org"
        self.initial_path = "/lepidoptera"
        self.first_page = 0
        self.last_page = 145
        self.batch_size = 3
        self.max_workers = 50
        self.page_timeout = 120
        self.image_timeout = 30
        self.ignore_image_regex = r"^(imgs10|.*(boimobileapp|butterfliesofurbangreeneries|webheader|headerlogo|WPA-[IVX]+).*)\.(png|jpg|jpeg)$"
        self.skip_downloaded_species = False
        self.early_stage_suffix = '-early'
        self.new_species_url = f"{self.website_url}/history-of-moth-species-pages-new"

    def log_header(self):
        return f"[ {threading.current_thread().name:24} ]  "

    def download_image(self, img_url, output_dir):
        try:
            img_data = requests.get(img_url, timeout=self.image_timeout).content
            img_name = img_url.split("/")[-1]
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
            if parent.get(prop) == value:
                return True
        return False
        
    def scrape_images(self, url, output_dir):
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
                    # larval host plants photo
                    # print (f"Ignoring larval host plants image {img_url}")
                    continue
                class_suffix = ''
                if self.has_parent_with_prop(img, 'id', 'early', 6):
                    # early stages
                    # print (f"Found early stages image {img_url}")
                    class_suffix = self.early_stage_suffix
                if img_url:
                    img_url = urljoin(url, img_url)
                    img_name = img_url.split("/")[-1]
                    if re.search(self.ignore_image_regex, img_name):
                        continue
                    if os.path.exists(f"{output_dir}{class_suffix}/{img_name}"):
                        already_downloaded_cnt = already_downloaded_cnt + 1
                        continue
                    if not os.path.exists(output_dir+class_suffix):
                        os.makedirs(output_dir+class_suffix)
                    if self.download_image(img_url, output_dir+class_suffix):
                        success_cnt = success_cnt + 1
                    else:
                        failure_cnt = failure_cnt + 1
            if success_cnt > 0:
                print(f"{self.log_header()}      Downloaded {success_cnt}(+{already_downloaded_cnt}) / {success_cnt+already_downloaded_cnt+failure_cnt} image(s) in {output_dir}")
            return True
        except Exception as e:
            print(f"{self.log_header()}{e}")
            return False

    def crawl(self, base_url, root, output_dir):
        try:
            url = urljoin(base_url, root)
            print(f"{self.log_header()}Crawling URL: {url}")
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
            response = requests.get(url, timeout=self.page_timeout)
            soup = BeautifulSoup(response.text, 'html.parser')
            img_tags = soup.find_all('img')
            for img in img_tags:
                img_url = img.get('src')
                if img_url:
                    img_url = urljoin(url, img_url)
                    # print (f"Found image {img_url}")
                    img_name = img_url.split("/")[-1]
                    if re.search(self.ignore_image_regex, img_name):
                        continue
                    species_path = img.parent.parent.get('href')
                    species_dir = output_dir+species_path.lower()
                    if not (self.skip_downloaded_species and os.path.exists(species_dir) and os.path.isdir(species_dir) and os.listdir(species_dir)):
                        self.scrape_images(urljoin(base_url, species_path), species_dir)
            return True
        except Exception as e:
            print(f"{self.log_header()}{e}")
            return False

    def crawl_in_batch(self, batch_start):
        print(f"Starting batch from page {batch_start} on thread {threading.current_thread().name}")
        for page in range(batch_start, min(batch_start+self.batch_size, self.last_page+1), 1):
            self.crawl(self.website_url, f"{self.initial_path}?page={page}", f"{self.dataset_dir}/data")

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
                
                # Remove query parameters if any
                if "?" in href:
                    href = href.split("?")[0]
                    
                species_name = href.strip("/")
                
                if not species_name:
                    continue

                # Heuristic: Species pages on this site usually start with an Uppercase letter (e.g., /Genus-species)
                # Navigation links are usually lowercase.
                if not species_name[0].isupper():
                    continue
                    
                normalized_name = species_name.lower()
                
                # Filter out paths with deeper structure if it somehow passed
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


class INaturalistScraper:
    def __init__(self):
        self.dataset_dir = "insect-dataset/src/inaturalist.org"
        self.page_timeout = 120
        self.image_timeout = 30
        self.max_workers = 50

    def log_header(self):
        return f"[ {threading.current_thread().name:24} ]  "

    def check_image(self, file_path):
        try:
            with Image.open(file_path) as img:
                img.verify()
            return True
        except (IOError, SyntaxError):
            return False
        
    def download_image(self, img_url, output_dir, uuid):
        try:
            # print(f"{self.log_header()} Downloading {img_url} into {output_dir}")
            # print(f"{self.log_header()} Downloading {img_url.split("/")[-1].split("?")[0]} into {output_dir.split("/")[-1]}")
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
            img_name = img_url.split("/")[-1]
            img_path = os.path.join(output_dir, f"{uuid}-{img_name.split('?')[0]}")
            if Path(img_path).is_file() and self.check_image(img_path):
                # skipping, already downloaded
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
            # print(f"{self.log_header()}{e}")
            if not os.listdir(output_dir):
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
                # print(f"{self.log_header()}Processing {class_name} | taxon_id:{taxon_id}")
                response = self.get_observations(taxon_id, 1)
                for result in response.json().get("results", []):
                    suffixed_class_name = class_name
                    if "tags" in result and len([ t for t in result["tags"] if re.match(r"^.*(egg|eggs|larva|larvae|pupa|pupae).*$", t)]) > 0:
                        suffixed_class_name += "-early"
                    for observation in result.get("observation_photos", []):
                        if "photo" in observation:
                            img_url = re.sub(r"\bsquare\b", "medium", observation["photo"]["url"])
                            status = self.download_image(img_url, f"{self.dataset_dir}/{suffixed_class_name}", observation["uuid"])
                            success_cnt += 1 if status=='SUCCESS' else 0
                            failure_cnt += 1 if status=='FAILURE' else 0
                            exists_cnt += 1 if status=='EXISTS' else 0
                if success_cnt + failure_cnt + exists_cnt > 0:
                    print(f"{self.log_header()}Processed {class_name} | taxon_id:{taxon_id}")
                    print(f"{self.log_header()}SUCCESS: {success_cnt:5} | FAILURE: {failure_cnt:5} | EXISTS: {exists_cnt:5}")
            except Exception as ex:
                # print(f"{self.log_header()}{ex}")
                continue

    def scrape_multithread(self, class_names, batch_size, skip_existing_dir=False):
        # print(f"{self.log_header()}Starting scraping...")
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = [executor.submit(self.scrape, class_names[offset:(offset+batch_size)], skip_existing_dir) for offset in range(0, len(class_names), batch_size)]
            for future in futures:
                result = future.result()
                # print(f"{self.log_header()}Thread completed with result {result}")
        # print(f"{self.log_header()}Scraping completed")

    def scrape_new_data_v2(self, classes):
        if not os.path.exists(self.dataset_dir):
            os.makedirs(self.dataset_dir)
            
        scraped_class_cnt = len([ class_dir for class_dir in os.listdir(self.dataset_dir) ])
        initial_class_cnt = scraped_class_cnt
        successive_failure_cnt = 0
        for i in range(0, 10000):
            self.scrape_multithread(classes, 1, skip_existing_dir=True)
            new_scraped_class_cnt = len([ class_dir for class_dir in os.listdir(self.dataset_dir) ])
            if new_scraped_class_cnt - scraped_class_cnt > 0:
                scraped_class_cnt = new_scraped_class_cnt
                successive_failure_cnt = 0
            else:
                successive_failure_cnt += 1
                # print(f"{self.log_header()}Sleeping 10s")
                time.sleep(60)
                if successive_failure_cnt > 5:
                    break
            # print(f"{self.log_header()}{scraped_class_cnt - initial_class_cnt} new classes added")
        print(f"{self.log_header()}All completed")



if __name__ == "__main__":
    moths_of_india_scraper = MothsOfIndiaScraper()
    inaturalist_scraper = INaturalistScraper()

    print("Scraping Moths of India for new species...")
    
    new_species = moths_of_india_scraper.fetch_missing_species()
    print(new_species)

    species_json = load_json("species.json")
    moth_species_list = species_json['lepidoptera']['species'][1]['names']

    if new_species:
        moth_species_list = sorted(list(set(moth_species_list + new_species)))
        species_json['lepidoptera']['species'][1]['names'] = moth_species_list
        dump_json("species.json", species_json)
    
    print("Scraping INaturalist for new species images...")

    inaturalist_scraper.scrape_new_data_v2(new_species)
    
    print("Scraping Moths of India for new species images...")

    for species in new_species:
        if not os.path.exists(f"{moths_of_india_scraper.dataset_dir}/{species}"):
            species_url = f"https://www.mothsofindia.org/{species}"
            moths_of_india_scraper.scrape_images(species_url, f"{moths_of_india_scraper.dataset_dir}/{species}")