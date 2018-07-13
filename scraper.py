from requests import get
from requests.exceptions import RequestException
from contextlib import closing
from bs4 import BeautifulSoup


def simple_get(url):
    try:
        with closing(get(url, stream=True)) as resp:
            if is_good_response(resp):
                return resp.content
            else:
                return None

    except RequestException as e:
        log_error('Error during requests to {0} : {1}'.format(url, str(e)))
        return None


def is_good_response(resp):
    content_type = resp.headers['Content-Type'].lower()
    return resp.status_code == 200 and content_type is not None and content_type.find('html') > -1


def log_error(e):
    print(e)


def extract(url):
    raw_html = simple_get(url)
    html = BeautifulSoup(raw_html, 'html.parser')
    data = {}

    for i, tr in enumerate(html.find_all('tr')):
        if (tr.find('td') is not None):
            num = tr.find('td', class_='code').text
            num = num.replace('U+', '')
            num = [chr(int(x, 16)) for x in num.split()]
            num = ''.join(num)
            name = tr.find('td', class_='name').text
            data[num] = name
        else:
            continue


    return data
