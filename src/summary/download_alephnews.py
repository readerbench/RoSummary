from __future__ import annotations

import os

import tqdm
import json
import requests
from bs4 import BeautifulSoup
from multiprocessing import Pool


def get_info_article(url: str) -> dict:
    page = requests.get(url)
    content = BeautifulSoup(page.content, 'html.parser')
    main_article = content.find('div', class_='article-wrap')
    header = content.find('div', class_='article-header')
    title = header.find('h1').text

    for div in main_article.find_all('div'):
        div.decompose()

    try:
        summary = [li.text.strip() for li in main_article.find('ul').find_all('li')]
    except AttributeError:
        summary = []

    for ul in main_article.find_all('ul'):
        ul.decompose()

    paragraphs = [paragraph.text.strip() for paragraph in main_article.find_all('p', recursive=False)]

    return {'url': url, 'title': title, 'summary': summary, 'paragraphs': paragraphs}


def get_list_url(url_base: str) -> list[str]:
    page = requests.get(url_base)
    main_content = BeautifulSoup(page.content, 'html.parser').find('main')

    return [news.find('a', href=True)['href'] for news in main_content.find('ul').find_all('li')]


def worker(config: tuple) -> None:
    path_base = '../../dataset/aleph_news/data/'
    news = []
    for url in tqdm.tqdm(config[1]):
        news.append(get_info_article(url))
    # news = [get_info_article(url) for url in config[1]]

    if not os.path.exists(path_base):
        os.makedirs(path_base)

    with open(f'{path_base}{config[0]}.json', 'w+') as output_file:
        json.dump(news, output_file, indent=4, ensure_ascii=False)


def create_links_list(total_number: int) -> None:
    url_base = 'https://alephnews.ro/ultimele-stiri/page/'
    path_dataset_aleph_news = '../../dataset/aleph_news'
    save_news_link = dict()

    for i in tqdm.tqdm(range(1, total_number + 1)):
        url_news = f'{url_base}{i}/'
        urls = get_list_url(url_news)
        save_news_link[i] = urls

    if not os.path.exists(path_dataset_aleph_news):
        os.makedirs(path_dataset_aleph_news)

    with open(os.path.join(path_dataset_aleph_news, 'links_alephnews.json'), 'w+') as output_file:
        json.dump(save_news_link, output_file, ensure_ascii=False, indent=4)


def create_dataset_aleph() -> None:
    path_json_link = '../../dataset/aleph_news/links_alephnews.json'
    path_json_link_missing = '../../dataset/aleph_news/missing_links.json'
    path_json_link = path_json_link_missing # path_json_link
    links = json.load(open(path_json_link, 'r'))

    configs = [(int(no_links), links) for no_links, links in links.items()]

    with Pool(40) as pool:
        pool.map_async(worker, configs)
        pool.close()
        pool.join()


def check_integrity() -> None:
    path_json_link = '../../dataset/aleph_news/links_alephnews.json'
    config = json.load(open(path_json_link, 'r'))
    path_dataset_aleph_news = '../../dataset/aleph_news/data'
    missing_links = dict()

    for index, list_links in config.items():
        path_downloader = os.path.join(path_dataset_aleph_news, f'{index}.json')

        if not os.path.exists(path_downloader):
            print(f'Missing: {path_downloader}')
            missing_links[index] = list_links
            continue

        with open(path_downloader, 'r') as file_input:
            data = json.load(file_input)

        if len(list_links) != len(data):
            print(f'For {path_downloader} is not ok!')

    with open('../../dataset/aleph_news/missing_links.json', 'w+') as file_missing:
        json.dump(missing_links, file_missing, indent=4, ensure_ascii=False)


if __name__ == '__main__':
    # create_links_list(2867)
    # create_dataset_aleph()
    check_integrity()
