"""
3.1. Этап 1. Формирование экспериментального корпуса текстов
Задача: Составить репрезентативный корпус современных русскоязычных новостных текстов.

Указания к выполнению:
Источники данных:
Рекомендуется использовать материалы новостных агентств и порталов:
ria.ru, tass.ru, lenta.ru, meduza.io, kommersant.ru и др.

💡 Инклюзивное задание:
Если вы являетесь представителем народов РФ (татары, башкиры, удмурты, чуваш, марийцы и др.), постарайтесь найти новостные сайты на своём родном языке. Ваша работа поможет развивать и сохранять языковое многообразие России.

Структура данных (обязательно для каждой статьи):

Заголовок
Основной текст
Дата публикации
URL-адрес
Категория/рубрика (при наличии)
Инструментарий:

requests + BeautifulSoup4 — для статических страниц
selenium — для динамических ресурсов
Требования к корпусу:

Общий объём: не менее 50 000 слов
Формат хранения: JSONL (каждая строка — отдельный JSON-объект со статьёй)
"""

import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
import time
import json
import re
from tqdm import tqdm
from dateutil import parser as dtparser
import dateparser
from text_cleaner import clean_text
# Заголовки для HTTP-запросов (чтобы сайт не блокировал как бота)
HEADERS = {
    "User-Agent": "Mozilla/5.0 (compatible; NewsCorpusBuilder/1.0; +https://example.org/bot)"
}

#настройки
TARGET_WORDS = 55_000#когда суммарное количество слов достигнет этого значения — процесс остановится
PER_SITE_DELAY = 1.0#задержка между запросами к одному сайту (в секундах)
REQUEST_TIMEOUT = 15#таймаут для HTTP-запросов
MIN_WORDS_IN_ARTICLE = 100#минимальное допустимое количество слов в статье

"""
Безопасно делает запрос к странице. Возвращает HTML-текст или None при ошибке
Параметры:
uro - адрес страницы
"""
def get_html(url):
    try:
        r = requests.get(url, headers=HEADERS, timeout=REQUEST_TIMEOUT)
        r.raise_for_status()#выброс исключения если статус не 200
        return r.text
    except Exception as e:
        print(f"[ERR] request {url}: {e}")
        return None

"""
Извлечение заголовка из html
Параметры:
soup - распарсенный html
"""
def extract_title(soup):
    meta = soup.select_one('meta[property="og:title"], meta[name="og:title"]')
    if meta and meta.get("content"):
        return meta["content"].strip()
    h1 = soup.find("h1")
    if h1 and h1.get_text(strip=True):
        return h1.get_text(strip=True)
    title_tag = soup.title
    if title_tag:
        return title_tag.get_text(strip=True)
    return None

"""
Извлекает дату публикации статьи, если она указана в мета-тегах или тексте
Параметры:
soup - распарсенный html
"""
def extract_date(soup):
    meta = soup.select_one(
        'meta[property="article:published_time"], meta[name="article:published_time"], meta[itemprop="datePublished"]'
    )
    if meta and meta.get("content"):
        try:
            return dateparser.parse(meta["content"])
        except:
            pass

    time_tag = soup.find("time")
    if time_tag:
        t = time_tag.get("datetime") or time_tag.get_text(strip=True)
        try:
            return dateparser.parse(t)
        except:
            pass

    #поиск даты в тексте
    text = soup.get_text(separator=" ", strip=True)
    m = re.search(r'(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2})', text)
    if m:
        try:
            return dtparser.parse(m.group(1))
        except:
            pass

    return None

"""
Извлекает категории статьи(если есть)
Параметры:
soup - распарсенный html
"""
def extract_category(soup):
    meta = soup.select_one('meta[property="article:section"], meta[name="article:section"]')
    if meta and meta.get("content"):
        return meta["content"].strip()

    crumbs = soup.select_one(".breadcrumb, .breadcrumbs")
    if crumbs:
        txt = crumbs.get_text(" ", strip=True)
        return txt
    return None

"""
Основной текст статьи: очищает HTML и извлекает текст из <article> или <p>.
"""
def extract_text(soup):
    result = ""
    #удаляем элементы, не содержащие текст статьи
    for s in soup(["script", "style", "aside", "header", "footer", "nav"]):
        s.decompose()

    #пробуем найти <article>
    article = soup.find("article")
    paragraphs = []
    if article:
        paragraphs = [p.get_text(" ", strip=True) for p in article.find_all("p")]
        paragraphs = [p for p in paragraphs if len(p.split()) >= 3]
        if paragraphs:
            result = "\n\n".join(paragraphs)

    #пробуем блоки с классами article, content, post, body
    candidates = soup.find_all(
        lambda tag: tag.name in ("div", "section")
        and tag.get("class")
        and any(
            re.search(r"(article|content|text|post|body)", c, re.I)
            for c in " ".join(tag.get("class"))
        )
    )
    for c in candidates:
        ps = [p.get_text(" ", strip=True) for p in c.find_all("p")]
        ps = [p for p in ps if len(p.split()) >= 3]
        if ps:
            result = "\n\n".join(ps)

    #фолбэк — взять все <p> на странице
    ps = [p.get_text(" ", strip=True) for p in soup.find_all("p")]
    ps = [p for p in ps if len(p.split()) >= 3]
    if ps:
        result = "\n\n".join(ps)
    result = clean_text(result, True, True)
    return result

"""
Преобразует относительные ссылки в абсолютные
параметры:
base - домен
link - путь к статье без домена
"""
def canonicalize_url(base, link):
    return urljoin(base, link)

"""
извлекает данные одной статьи: заголовок, текст, дату, категорию и ссылку
параметры:
url - адрес страницы
"""
def parse_article(url):
    html = get_html(url)
    if not html:
        return None

    soup = BeautifulSoup(html, "html.parser")

    title = extract_title(soup) or ""
    text = extract_text(soup) or ""
    if not text.strip():
        return None

    date = extract_date(soup)
    category = extract_category(soup)

    return {
        "title": title,
        "text": text,
        "date": date.isoformat() if date else None,
        "url": url,
        "category": category,
    }

"""
Парсит индексную страницу и собирает ссылки на статьи
"""
def parse_index_for_links(index_url, allowed_domain=None):
    html = get_html(index_url)
    if not html:
        return []

    soup = BeautifulSoup(html, "html.parser")
    links = set()

    for a in soup.find_all("a", href=True):
        href = canonicalize_url(index_url, a["href"])
        p = urlparse(href)

        if p.scheme not in ("http", "https"):
            continue

        #фильтрация по домену
        if allowed_domain and p.netloc and allowed_domain not in p.netloc:
            continue

        #исключаем ссылки на изображения, pdf и архивы
        if re.search(r"\.(jpg|jpeg|png|gif|pdf|zip)$", p.path, re.I):
            continue

        links.add(href)

    return list(links)

"""
Основная функция парсера:
1. Собирает все ссылки с заданных сайтов
2. Парсит статьи
3. Сохраняет результат в JSONL
Останавливается, когда суммарное количество слов >= TARGET_WORDS
"""
def main(seed_indexes, out_jsonl="corpus.jsonl"):
    total_articles = 0
    total_words = 0#суммарное количество слов во всех сохранённых статьях
    session_last_request = {}#отслеживает, когда последний раз запрашивали домен

    with open(out_jsonl, "a", encoding="utf-8") as fout:
        #сначала собираем все возможные ссылки на статьи
        candidate_links = []
        for idx in seed_indexes:
            domain = urlparse(idx).netloc
            #задержка между запросами к одному домену
            last = session_last_request.get(domain)
            if last:
                diff = time.time() - last
                if diff < PER_SITE_DELAY:
                    time.sleep(PER_SITE_DELAY - diff)

            links = parse_index_for_links(idx, allowed_domain=domain)
            session_last_request[domain] = time.time()
            candidate_links.extend(links)

        #убираем дубликаты по URL
        candidate_links = list(dict.fromkeys(candidate_links))

        #проходим по каждой ссылке
        for link in tqdm(candidate_links, desc="Parsing links"):
            #если набрали нужное количество слов — выходим
            if total_words >= TARGET_WORDS:
                print(f"Набрано нужное количество слов - {total_words}")
                break

            domain = urlparse(link).netloc

            #соблюдаем задержку между запросами
            last = session_last_request.get(domain)
            if last:
                diff = time.time() - last
                if diff < PER_SITE_DELAY:
                    time.sleep(PER_SITE_DELAY - diff)

            parsed = parse_article(link)
            session_last_request[domain] = time.time()

            if not parsed:
                continue

            #считаем количество слов в статье
            word_count = len(parsed["text"].split())

            #если слов мало, то не сохраняем статью
            if word_count < MIN_WORDS_IN_ARTICLE:
                continue

            # сохраняем статью в JSONL
            fout.write(json.dumps(parsed, ensure_ascii=False) + "\n")
            fout.flush()

            total_articles += 1
            total_words += word_count

            #сообщение о прогрессе
            print(f"Сохранено: {parsed['url']} ({word_count} слов)")

            # Если после добавления этой статьи мы достигли цели — останавливаемся
            if total_words >= TARGET_WORDS:
                print(f"Набрано нужное количество слов - {total_words}")
                break

if __name__ == "__main__":
    # Пример стартовых страниц (новостные сайты)
    seeds = [
        "https://ria.ru/",
        "https://lenta.ru/",
        "https://tass.ru/",
        "https://meduza.io/",
        "https://www.kommersant.ru/",
    ]
    main(seeds, out_jsonl="corpus.jsonl")
