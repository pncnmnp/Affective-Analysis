from gutenberg.acquire import load_etext
from gutenberg.cleanup import strip_headers
from gutenberg.query import get_metadata

import json
import os
import glob

BOOK_ID_PATH = "./gutenberg_book_ids.json"
GUTENBERG_META_PATH = "./gutenberg-metadata.json"


def get_json_file(filename):
	return json.load(open(filename))

def get_meta(book_ids, all_meta):
	meta = list()
	for book_id in book_ids:
		meta.append(all_meta[str(book_id)]["title"] + [str(book_id)])
		meta.append(all_meta[str(book_id)]["subject"])

	with open("meta.json", 'w', encoding='utf-8') as f:
		json.dump(meta, f)

def download(book_ids, all_meta, base_directory="./books/"):
	for book_id in book_ids:
		if os.path.isfile(base_directory+str(book_id)+".json"):
			print("File already downloaded: {}".format(str(book_id)+".json"))
			continue
		else:
			try:
				meta = all_meta[str(book_id)]
				text = strip_headers(load_etext(book_id))
				meta["text"] = text

				with open(base_directory+str(book_id)+".json", 'w', encoding='utf-8') as f:
					json.dump(meta, f)

				print("DOWNLOADED: {}".format(str(book_id)+".json"))

			except:
				print("COULD NOT DOWNLOAD: {}".format(str(book_id)+".json"))

if __name__ == "__main__":
	# DOWNLOAD BOOKS
	# book_ids = get_json_file(BOOK_ID_PATH)
	# all_meta = get_json_file(GUTENBERG_META_PATH)

	# download([59183], all_meta)

	# DOWNLOAD META
	all_meta = get_json_file(GUTENBERG_META_PATH)
	downloaded_books = sorted([int(file.replace("./stats/", "").replace(".csv", "")) for file in glob.glob("./stats/*.csv")])
	get_meta(book_ids=downloaded_books, all_meta=all_meta)