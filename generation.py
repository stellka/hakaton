import random
import pandas as pd


# Генерация случайных URL
def generate_random_url(phishing=False):
    domain_list = ['example', 'test', 'website', 'domain', 'myblog', 'shop']
    tld_list = ['.com', '.org', '.net', '.info', '.xyz']

    # Для фишинговых сайтов добавляем случайные паттерны
    if phishing:
        path_list = ['login', 'signin', 'secure', 'verify', 'account']
        random_path = '/' + random.choice(path_list) + '?id=' + str(random.randint(1000, 9999))
        return f'http://{random.choice(domain_list)}{random.choice(tld_list)}{random_path}'
    else:
        return f'http://{random.choice(domain_list)}{random.choice(tld_list)}'


# Генерация данных
def generate_data(num_safe=500, num_phishing=500):
    data = []
    for _ in range(num_safe):
        data.append([generate_random_url(phishing=False), 0])
    for _ in range(num_phishing):
        data.append([generate_random_url(phishing=True), 1])

    # Создание DataFrame
    df = pd.DataFrame(data, columns=['url', 'result'])
    return df


# Генерация 1000 записей (500 безопасных, 500 фишинговых)
synthetic_data = generate_data(500, 500)
synthetic_data.to_csv('synthetic_data.csv', index=False)
print("Синтетические данные сохранены в 'synthetic_data.csv'")
