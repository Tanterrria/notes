# Решение для предсказания оттока сотрудников

## 1. Модель машинного обучения
* **Выбранная модель:** XGBoost (eXtreme Gradient Boosting)
* **Тип задачи:** Бинарная классификация (уволится/не уволится)

### 1.1 Преимущества XGBoost для данной задачи
* Высокая точность предсказаний
* Встроенная регуляризация для предотвращения переобучения
* Возможность работы с пропущенными значениями
* Встроенные инструменты для анализа важности признаков
* Эффективная работа с числовыми признаками

### 1.2 Ключевые гиперпараметры XGBoost
* `max_depth`: максимальная глубина дерева (рекомендуется 3-6)
* `learning_rate`: скорость обучения (рекомендуется 0.01-0.1)
* `n_estimators`: количество деревьев (рекомендуется 100-500)
* `subsample`: доля данных для обучения каждого дерева (0.6-0.8)
* `colsample_bytree`: доля признаков для каждого дерева (0.6-0.8)
* `reg_alpha`: L1 регуляризация (0-1)
* `reg_lambda`: L2 регуляризация (0-1)

Пример значений для гиперпараметров:
```python
param_grid = {
    'max_depth': [3, 4, 5, 6],
    'learning_rate': [0.01, 0.05, 0.1],
    'n_estimators': [100, 200, 300],
    'subsample': [0.6, 0.7, 0.8],
    'colsample_bytree': [0.6, 0.7, 0.8],
    'reg_alpha': [0, 0.1, 1],
    'reg_lambda': [0, 0.1, 1]
}
```

В машинном обучении мы пытаемся найти функцию, которая минимизировала бы функцию потерь, то есть чтобы наборы данных трейна Х и предсказания У были близки друг к другу. Но если мы не будем следить за состоянием, насколько они близки друг к другу, то придем к переобучению (предсказание хорошо отрабатывает на тренировочных данных и никак на валидационных, слишком хорошо под тест подстроились). Регуляризация - добавление штрафа к функции потерь.
L1 регуляризация - избавляет нас от незначимых признаков.
L2 регуляризация - борется с коррелирующими признаками.
(не забываем, что все хорошо в меру - не перебарщиваем)

## 2. Ключевые признаки для модели

### 2.1 Метрики посещаемости
* `cnt_day_abs` - Прогул
* `cnt_day_bol` - Больничный
* `cnt_day_hol` - Отпуск

### 2.2 Метрики эффективности
* `emp_eff` - Общая производительность сотрудника
* `emp_chron_eff` - Хронометражная производительность сотрудника
* `dm_ratio_eff` - Доля производительности к ДМ
* `off_ratio_eff` - Доля производительности сотрудника к средней производительности офиса

### 2.3 Метрики дохода
* `fact_doh_steet` - Фактический доход сотрудника за месяц (без отпускных и больничных) с учетом отработанного времени
* `emp_dm_ratio` - Доля дохода сотрудника от дохода ДМ
* `emp_off_ratio` - Доля дохода сотрудника от среднего/целевого дохода по офису
* `emp_reg_ratio` - Доля дохода сотрудника от среднего/целевого дохода на аналогичной должности по региону
* `reg_off_ratio` - Доля среднего дохода офиса от среднего/целевого дохода на аналогичной должности по региону

### 2.4 Метрики рабочего времени
* `overtime_wh` - Часов переработано/недоработано

### 2.5 Метрики нарушений
* `vio_frod` - Количество фродовых отклонений
* `vio_sales` - Количество отклонений

## 3. Предобработка данных
### Нормализация числовых признаков

Нормализация - это процесс преобразования числовых признаков к единому масштабу. Это важно, потому что:
* XGBoost чувствителен к масштабу признаков
* Признаки в разных масштабах могут влиять на модель по-разному
* Ускоряет процесс обучения

#### Основные методы нормализации:

1. **Min-Max Scaling (Масштабирование в диапазон [0,1])**
   * Преобразует данные в диапазон [0,1]
   * Сохраняет распределение
   * Чувствителен к выбросам
   * Формула: 
     ```
     X_scaled = (X - X_min) / (X_max - X_min)
     ```

2. **Standard Scaling (Z-score normalization)**
   * Преобразует данные к среднему 0 и стандартному отклонению 1
   * Хорошо работает с нормальным распределением
   * Менее чувствителен к выбросам
   * Формула:
     ```
     X_scaled = (X - μ) / σ
     ```
     где:
     * μ - среднее значение
     * σ - стандартное отклонение

3. **Robust Scaling**
   * Использует медиану и межквартильный размах
   * Устойчив к выбросам
   * Хорошо подходит для данных с аномалиями
   * Формула:
     ```
     X_scaled = (X - median) / (Q3 - Q1)
     ```
     где:
     * median - медиана
     * Q1 - первый квартиль
     * Q3 - третий квартиль

#### Применение к нашим данным:

1. **Метрики посещаемости:**
   * Min-Max Scaling
   * Признаки: `cnt_day_abs`, `cnt_day_bol`, `cnt_day_hol`
   * Причина: известные границы (0 до максимального количества дней)

2. **Метрики эффективности:**
   * Standard Scaling
   * Признаки: `emp_eff`, `emp_chron_eff`, `dm_ratio_eff`, `off_ratio_eff`
   * Причина: ожидается нормальное распределение

3. **Метрики дохода:**
   * Robust Scaling
   * Признаки: `fact_doh_steet`, `emp_dm_ratio`, `emp_off_ratio`, `emp_reg_ratio`
   * Причина: возможны выбросы в данных

#### Реализация:

```python
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler
from sklearn.compose import ColumnTransformer

# Определяем группы признаков
attendance_features = ['cnt_day_abs', 'cnt_day_bol', 'cnt_day_hol']
efficiency_features = ['emp_eff', 'emp_chron_eff', 'dm_ratio_eff', 'off_ratio_eff']
income_features = ['fact_doh_steet', 'emp_dm_ratio', 'emp_off_ratio', 'emp_reg_ratio']

# Создаем трансформер
preprocessor = ColumnTransformer(
    transformers=[
        ('attendance', MinMaxScaler(), attendance_features),
        ('efficiency', StandardScaler(), efficiency_features),
        ('income', RobustScaler(), income_features)
    ]
)

# Применяем нормализацию
X_scaled = preprocessor.fit_transform(X)
```

#### Рекомендации:
* Сохранять скейлеры для использования в production (короче. Мы вот на определенные параметры нормализовали, среднее посчитали, минимальные и макисмальные значения. Захотели получить предсказание, а там другие значения - а так нельзя! Используем те же параметры нормализации, какие мы использовали на трейне!)
* Проверять распределение признаков до и после нормализации:
  * Визуальная проверка с помощью графиков:
    ```python
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    # Создаем графики для каждого признака
    def plot_distribution(original, scaled, feature_name):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        
        # Распределение до нормализации
        sns.histplot(original, ax=ax1)
        ax1.set_title(f'{feature_name} до нормализации')
        
        # Распределение после нормализации
        sns.histplot(scaled, ax=ax2)
        ax2.set_title(f'{feature_name} после нормализации')
        
        plt.tight_layout()
        plt.show()
    
    # Проверяем статистики
    def print_statistics(original, scaled, feature_name):
        print(f'\nСтатистики для {feature_name}:')
        print('До нормализации:')
        print(f'Среднее: {original.mean():.2f}')
        print(f'Стд. отклонение: {original.std():.2f}')
        print(f'Мин: {original.min():.2f}')
        print(f'Макс: {original.max():.2f}')
        
        print('\nПосле нормализации:')
        print(f'Среднее: {scaled.mean():.2f}')
        print(f'Стд. отклонение: {scaled.std():.2f}')
        print(f'Мин: {scaled.min():.2f}')
        print(f'Макс: {scaled.max():.2f}')
    ```
  * Что проверять:
    * Для MinMax Scaling:
      - Значения должны быть в диапазоне [0,1]
      - Форма распределения должна сохраниться
      - Относительные различия между значениями должны остаться
    * Для Standard Scaling:
      - Среднее должно быть близко к 0
      - Стандартное отклонение должно быть близко к 1
      - Распределение должно остаться нормальным
    * Для Robust Scaling:
      - Медиана должна быть близка к 0
      - Межквартильный размах должен быть близок к 1
      - Выбросы должны быть менее заметны
  * Когда бить тревогу:
    * Если распределение сильно исказилось
    * Если появились неожиданные выбросы
    * Если масштаб не соответствует ожидаемому
    * Если относительные различия между значениями изменились


### Обработка пропущенных значений

#### Простые способы заполнения пропусков:

1. **Заполнение нулями**
   * Для метрик посещаемости (`cnt_day_abs`, `cnt_day_bol`, `cnt_day_hol`)
   * Для метрик нарушений (`vio_frod`, `vio_sales`)
   * Логика: если данных нет, значит событий не было

2. **Заполнение средним значением**
   * Для метрик эффективности (`emp_eff`, `emp_chron_eff`)
   * Для метрик дохода (`fact_doh_steet`, `emp_dm_ratio`)
   * Логика: заменяем пропуск на среднее значение по всем сотрудникам

3. **Заполнение медианой**
   * Для метрик с выбросами
   * Логика: медиана меньше чувствительна к выбросам, чем среднее

#### Реализация:
```python
# Заполнение нулями
df = df.fillna(0)

# Заполнение средним
df = df.fillna(df.mean())

# Заполнение медианой
df = df.fillna(df.median())
```

#### Рекомендации:
* Проверять количество пропусков перед заполнением
* Выбирать метод заполнения в зависимости от типа данных
* Не забывать сохранять параметры заполнения для production


### Балансировка классов

#### Почему это важно:
* В данных обычно больше сотрудников, которые не увольняются
* Модель может "забыть" про меньший класс
* Нужно для корректного предсказания увольнений

#### Простые способы балансировки:

1. **Взвешивание классов**
   * Даем больший вес классу увольнений
   * Простой способ, не требует изменения данных
   * Реализация:
     ```python
     # В XGBoost
     model = XGBClassifier(scale_pos_weight=10)  # если увольняется 1/10 сотрудников
     ```

2. **Undersampling**
   * Уменьшаем количество примеров большего класса
   * Берем случайную выборку из "не увольнений"
   * Реализация:
     ```python
     from imblearn.under_sampling import RandomUnderSampler
     
     rus = RandomUnderSampler(random_state=42)
     X_balanced, y_balanced = rus.fit_resample(X, y)
     ```

3. **Oversampling**
   * Увеличиваем количество примеров меньшего класса
   * Дублируем примеры увольнений
   * Реализация:
     ```python
     from imblearn.over_sampling import RandomOverSampler
     
     ros = RandomOverSampler(random_state=42)
     X_balanced, y_balanced = ros.fit_resample(X, y)
     ```

#### Рекомендации:
* Начинать с простого взвешивания классов
* Если не помогает - пробовать oversampling
* Undersampling использовать только если много данных
* Проверять метрики на несбалансированных данных

### Обработка выбросов в метриках эффективности и дохода

#### Почему это важно:
* Выбросы могут искажать результаты модели
* Особенно критично для метрик эффективности и дохода
* Могут быть как реальными данными, так и ошибками

#### Простые способы обработки:

1. **Обрезка (Clipping)**
   * Устанавливаем границы для значений
   * Все, что выходит за границы, приравниваем к границе
   * Реализация:
     ```python
     # Для метрик эффективности
     df['emp_eff'] = df['emp_eff'].clip(lower=0, upper=200)  # максимум 200%
     
     # Для метрик дохода
     df['fact_doh_steet'] = df['fact_doh_steet'].clip(
         lower=df['fact_doh_steet'].quantile(0.01),  # нижний 1%
         upper=df['fact_doh_steet'].quantile(0.99)   # верхний 1%
     )
     ```

2. **Удаление выбросов**
   * Удаляем значения, выходящие за границы
   * Используем межквартильный размах (IQR)
   * Реализация:
     ```python
     def remove_outliers(df, column):
         Q1 = df[column].quantile(0.25)
         Q3 = df[column].quantile(0.75)
         IQR = Q3 - Q1
         
         lower_bound = Q1 - 1.5 * IQR
         upper_bound = Q3 + 1.5 * IQR
         
         return df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]
     
     # Применяем к метрикам
     df = remove_outliers(df, 'emp_eff')
     df = remove_outliers(df, 'fact_doh_steet')
     ```

#### Рекомендации:
* Для метрик эффективности:
  * Использовать обрезку (не может быть отрицательной)
  * Установить разумный верхний предел (например, 200%)
* Для метрик дохода:
  * Использовать процентили для определения границ
  * Учитывать должность и регион
* Всегда проверять:
  * Сколько данных будет изменено/удалено
  * Как это повлияет на распределение
  * Есть ли бизнес-смысл в выбросах

Пример визуализации выбросов:
```python
import seaborn as sns
import matplotlib.pyplot as plt

def plot_outliers(df, column):
    plt.figure(figsize=(10, 6))
    sns.boxplot(x=df[column])
    plt.title(f'Распределение {column} с выбросами')
    plt.show()
    
    # После обработки выбросов
    df_cleaned = remove_outliers(df, column)
    plt.figure(figsize=(10, 6))
    sns.boxplot(x=df_cleaned[column])
    plt.title(f'Распределение {column} после обработки выбросов')
    plt.show()
```

## 4. Метрики для оценки модели
* Precision и Recall
* ROC-AUC
* F1-score

Примеры хороших значений метрик:
```python
print("Хорошие значения метрик:")
print("ROC-AUC: > 0.8")
print("F1-score: > 0.7")
print("Precision: > 0.6")
print("Recall: > 0.7")
```

Пример использования гридсёрча:
``` python
from sklearn.model_selection import GridSearchCV

param_grid = {
    'reg_alpha': [0, 0.1, 1, 10],
    'reg_lambda': [0, 0.1, 1, 10]
}

grid_search = GridSearchCV(XGBClassifier(), param_grid, cv=5)
grid_search.fit(X, y)
```

   Улучшенный вариант с кросс-валидацией (мы создаем несколько наборов данных, ):
``` python
from sklearn.model_selection import StratifiedKFold

# Создаем объект кросс-валидации
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Используем в GridSearchCV
grid_search = GridSearchCV(
    XGBClassifier(),
    param_grid,
    cv=skf,  # Используем стратифицированную кросс-валидацию
    scoring='roc_auc'  # Метрика для оценки
)
```

Использование нескольких метрик
``` python
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.metrics import make_scorer, roc_auc_score, f1_score

# Создаем объект кросс-валидации
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Расширенный набор метрик
scoring = {
    'roc_auc': 'roc_auc',
    'f1': make_scorer(f1_score),
    'precision': 'precision',
    'recall': 'recall',
    'average_precision': 'average_precision'  # PR-AUC для несбалансированных данных
}

# Настройка GridSearchCV
grid_search = GridSearchCV(
    XGBClassifier(),
    param_grid,
    cv=skf,
    scoring=scoring,
    refit='roc_auc',
    n_jobs=-1  # Использование всех доступных ядер
)

# Обучение
grid_search.fit(X, y)

# Анализ результатов
print("Лучшие параметры:", grid_search.best_params_)
print("\nРезультаты по всем метрикам:")
for metric in scoring.keys():
    print(f"{metric}: {grid_search.cv_results_[f'mean_test_{metric}'][grid_search.best_index_]:.3f}")
```

Интерпретация результатов:
* Если ROC-AUC высокий, а F1 низкий:
  * Модель хорошо различает классы
  * Но может быть проблема с порогом классификации (то есть если вероятность увольнения > 0,5 - он увольняется. 0,5 - это и есть порог классификации).
  * Нужно настроить порог для баланса precision/recall
* Если F1 высокий, а ROC-AUC низкий:
  * Модель хорошо работает на текущем пороге
  * Но может быть нестабильной
  * Нужно проверить на разных порогах
* Если оба показателя высокие:
  * Модель работает хорошо
  * Можно использовать для предсказаний

Основная метрика: **ROC-AUC**
* Хорошо работает с несбалансированными данными
* Учитывает все возможные пороги
Дополнительные метрики:
* F1-score для баланса ошибок
* Precision для минимизации ложных тревог
* Recall для минимизации пропущенных увольнений

### Анализ результатов GridSearch для определения промежуточных значений

#### Методы анализа результатов поиска:

1. **Визуализация результатов поиска**
   ```python
   import matplotlib.pyplot as plt
   import seaborn as sns
   
   def plot_grid_search_results(grid_search, param_name):
       # Получаем результаты
       results = grid_search.cv_results_
       param_values = results[f'param_{param_name}'].data
       mean_scores = results['mean_test_score']
       
       # Строим график
       plt.figure(figsize=(10, 6))
       plt.plot(param_values, mean_scores, 'o-')
       plt.xlabel(param_name)
       plt.ylabel('ROC-AUC')
       plt.title(f'Зависимость ROC-AUC от {param_name}')
       plt.grid(True)
       plt.show()
   
   # Пример использования
   plot_grid_search_results(grid_search, 'max_depth')
   ```

2. **Анализ разрыва между значениями**
   ```python
   def analyze_param_gaps(grid_search, param_name):
       results = grid_search.cv_results_
       param_values = results[f'param_{param_name}'].data
       mean_scores = results['mean_test_score']
       
       # Сортируем по значениям параметра
       sorted_indices = np.argsort(param_values)
       param_values = param_values[sorted_indices]
       mean_scores = mean_scores[sorted_indices]
       
       # Находим разрывы в производительности
       score_diffs = np.diff(mean_scores)
       significant_gaps = np.where(np.abs(score_diffs) > 0.01)[0]  # порог 0.01
       
       print(f"Значительные изменения в {param_name}:")
       for gap in significant_gaps:
           print(f"Между {param_values[gap]} и {param_values[gap+1]}: "
                 f"изменение ROC-AUC на {score_diffs[gap]:.3f}")
   ```

3. **Проверка стабильности результатов**
   ```python
   def check_param_stability(grid_search, param_name):
       results = grid_search.cv_results_
       param_values = results[f'param_{param_name}'].data
       std_scores = results['std_test_score']
       
       # Находим параметры с высокой стабильностью
       stable_params = np.where(std_scores < 0.05)[0]  # порог 0.05
       
       print(f"Стабильные значения {param_name}:")
       for idx in stable_params:
           print(f"{param_name} = {param_values[idx]}: "
                 f"std = {std_scores[idx]:.3f}")
   ```

#### Когда добавлять промежуточные значения в params гиперпараметр:

1. **Если на графике виден резкий скачок**
   * Между двумя значениями параметра большой разрыв в производительности
   * Нужно добавить значения между этими точками

2. **Если есть нестабильные результаты**
   * Большое стандартное отклонение для некоторых значений
   * Стоит добавить больше значений вокруг нестабильных точек

3. **Если лучшие параметры на границе диапазона**
   * Лучшее значение параметра - минимальное или максимальное
   * Нужно расширить диапазон поиска

#### Пример использования:
```python
# После выполнения GridSearch
grid_search.fit(X, y)

# Анализируем результаты
plot_grid_search_results(grid_search, 'max_depth')
analyze_param_gaps(grid_search, 'max_depth')
check_param_stability(grid_search, 'max_depth')

# На основе анализа корректируем param_grid
param_grid = {
    'max_depth': [3, 4, 5, 6, 7],  # добавили 7
    'learning_rate': [0.01, 0.03, 0.05, 0.1],  # добавили 0.03
    'n_estimators': [100, 200, 300, 400]  # добавили 400
}
```
