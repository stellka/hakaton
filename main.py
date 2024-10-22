import pandas as pd
import re
from urllib.parse import urlparse
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Шаг 1: Загрузка данных
data = pd.read_csv('train.csv')


# Шаг 2: Функции для предобработки URL
def preprocess_url(url):
    # Если схема отсутствует, добавляем 'http://'
    if not url.startswith(('http://', 'https://')):
        url = 'http://' + url

    # Парсинг URL
    parsed_url = urlparse(url)

    # Обработка IP-адресов
    if re.match(r'\d+\.\d+\.\d+\.\d+', parsed_url.netloc):
        return "IP_" + parsed_url.netloc  # Обозначаем IP как отдельный признак

    # Оставляем домен и путь
    domain = parsed_url.netloc
    path = parsed_url.path if parsed_url.path else '/'

    # Возвращаем строку вида 'домен/путь'
    return domain + path


# Применение предобработки ко всем URL
data['url'] = data['url'].apply(preprocess_url)


# Добавление дополнительных признаков
def extract_features(url):
    parsed_url = urlparse(url)

    # Признак: длина URL
    url_length = len(url)

    # Признак: количество поддоменов
    subdomains = parsed_url.netloc.split('.')
    num_subdomains = len(subdomains) - 2 if len(subdomains) > 2 else 0

    # Признак: наличие IP в URL
    has_ip = 1 if re.match(r'\d+\.\d+\.\d+\.\d+', parsed_url.netloc) else 0

    # Признак: наличие спецсимволов (@, -)
    has_at_symbol = 1 if '@' in url else 0
    has_dash = 1 if '-' in url else 0

    return [url_length, num_subdomains, has_ip, has_at_symbol, has_dash]


# Применение функции для извлечения признаков
data['features'] = data['url'].apply(extract_features)
features_df = pd.DataFrame(data['features'].tolist(),
                           columns=['url_length', 'num_subdomains', 'has_ip', 'has_at_symbol', 'has_dash'])

# Шаг 3: Подготовка данных
X_text = data['url']  # Текстовые признаки (обработанные URL)
X_additional = features_df  # Дополнительные признаки
y = data['result']  # Метки (0 или 1)

# Преобразование URL в векторное представление с TfidfVectorizer
vectorizer = TfidfVectorizer(max_features=3000)  # Ограничиваем количество признаков для лучшей производительности
X_text_tfidf = vectorizer.fit_transform(X_text)

# Объединяем текстовые и дополнительные признаки
import scipy

X_combined = scipy.sparse.hstack([X_text_tfidf, X_additional])

# Разделение на обучающую и тестовую выборки
X_train, X_test, y_train, y_test = train_test_split(X_combined, y, test_size=0.2, random_state=42)

# Шаг 4: Обучение модели Random Forest
model = RandomForestClassifier(n_estimators=200, random_state=42, class_weight='balanced')

# Подбор гиперпараметров с помощью GridSearchCV
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [10, 20, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

grid_search = GridSearchCV(model, param_grid, cv=5, scoring='accuracy', n_jobs=-1)
grid_search.fit(X_train, y_train)

# Лучшая модель
best_model = grid_search.best_estimator_

# Шаг 5: Оценка модели на тестовых данных
y_pred = best_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f'Точность модели: {accuracy:.2f}')


# Шаг 6: Предсказание для новых данных
def predict_phishing(urls):
    urls = urls.apply(preprocess_url)
    urls_features = urls.apply(extract_features)
    urls_features_df = pd.DataFrame(urls_features.tolist(),
                                    columns=['url_length', 'num_subdomains', 'has_ip', 'has_at_symbol', 'has_dash'])

    # Преобразование URL в векторное представление
    urls_tfidf = vectorizer.transform(urls)

    # Объединение текстовых и дополнительных признаков
    urls_combined = scipy.sparse.hstack([urls_tfidf, urls_features_df])

    # Предсказания
    predictions = best_model.predict(urls_combined)
    return predictions


# Загрузка тестового списка URL
test_data = pd.read_csv('test.csv')

# Прогнозирование
test_urls = test_data['url']
test_predictions = predict_phishing(test_urls)

# Формирование итогового DataFrame с колонками Id и Predicted
test_data['Id'] = test_data.index  # Создание колонки Id
test_data['Predicted'] = test_predictions  # Прогнозы

# Оставляем только необходимые колонки
result = test_data[['Id', 'Predicted']]

# Сохранение результатов в CSV файл
result.to_csv('sample_submission.csv', index=False)

print("Результаты сохранены в 'sample_submission.csv'")
