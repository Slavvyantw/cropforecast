
from tkinter import filedialog
import tkinter as tk
import matplotlib.pyplot as plt
import numpy as np
import math

def compute_seasonal_index(data, trend, seasonality, period=12):
    """
    Вычисление индекса сезонности.

    data: numpy array, исходный временной ряд x_t
    trend: numpy array, трендовая составляющая y_t
    seasonality: numpy array, сезонная составляющая s_t
    period: int, количество периодов в цикле (например, 12 для месячных данных)

    Возвращает: массив индексов сезонности.
    """
    # Остатки после удаления тренда
    detrended = data - trend

    # Группировка по периодам (например, по месяцам)
    grouped = {i: [] for i in range(period)}
    for i, value in enumerate(detrended):
        grouped[i % period].append(value)

    # Рассчитываем среднее по каждому периоду
    seasonal_index = np.array([np.mean(grouped[i]) for i in range(period)])

    # Нормализация индекса (среднее значение = 1)
    seasonal_index /= np.mean(seasonal_index)
    return seasonal_index
def load_file():
    # Создаем окно для выбора файла
    root = tk.Tk()
    root.withdraw()  # Скрыть главное окно

    # Открываем диалоговое окно для выбора файла
    file_path = filedialog.askopenfilename(title="Выберите файл", filetypes=[("Text files", "*.txt")])

    if file_path:
        # Если файл выбран, читаем его содержимое
        with open(file_path, 'r') as file:
            data = file.read().strip().split()
        
        # Преобразуем данные в список
        data = [float(num) for num in data]  # Убираем символы новой строки
        
        
        
        # Здесь можно обработать данные, например, преобразовать их в массив по месяцам
        return data
    else:
        print("Файл не выбран.")
        return None
def compute_residual_distribution(data, trend, seasonality, bins=7):
    """
    Вычисление распределения частоты остаточного члена по интервалам.

    data: numpy array, исходный временной ряд x_t
    trend: numpy array, трендовая составляющая y_t
    seasonality: numpy array, сезонная составляющая s_t
    bins: int, количество интервалов

    Возвращает:
    - intervals: массив интервалов
    - frequencies: массив частот попадания в интервалы
    """
    # Остаточный член
    residual = data - (trend + seasonality)

    # Определяем интервалы
    min_val, max_val = np.min(residual), np.max(residual)
    intervals = np.linspace(min_val, max_val, bins + 1)

    # Вычисляем частоты
    frequencies = np.histogram(residual, bins=intervals)[0]

    return intervals, frequencies, residual
def golden_section_search(f, a, b, tol=1e-5):
    # Золотое сечение
    gr = (math.sqrt(5) - 1) / 2
    
    # Начальные точки
    c = b - gr * (b - a)
    d = a + gr * (b - a)
    
    # Создаем список для хранения данных
    results = []
    
    while abs(a - b) > tol:
        if f(c) < f(d):
            b = d
        else:
            a = c
        
        # Сохраняем текущие данные
        current_lamda = (a + b) / 2
        current_mse = f(current_lamda)
        interval = (a, b)
        results.append((current_lamda, current_mse, interval))
        
        # Пересчитываем точки
        c = b - gr * (b - a)
        d = a + gr * (b - a)
    
    # Записываем результаты в файл
    with open("optimization_results.txt", "w") as file:
        file.write("Lambda\t\tMSE\t\tInterval\n")
        for lamda, mse, interval in results:
            file.write(f"{lamda:.6f}\t{mse:.6f}\t[{interval[0]:.6f}, {interval[1]:.6f}]\n")
    
    return (a + b) / 2
def exponential_weights(length, alpha):
    weights = np.array([alpha * (1 - alpha) ** (length - t) for t in range(1, length + 1)])
    return weights / weights.sum()
def wight_series(X,weight,n,m,k):
    X=X[:-12]
    dlina=(len(X))
    t=np.arange(1,dlina+1)
    weights =np.exp(-weight * (dlina - t))
    B = np.diag(weights)

    trend=np.vstack([t**i for i in range(n + 1)]).T
    season=[]
    for i in range(m+1):
        for j in range(1, k // 2+1 ):
            season.append(np.cos(2 * np.pi * j * t / k) * (t**i))  # Косинусные гармоники
        for j in range(1, k // 2):  # Обратите внимание на диапазон j для синусных гармоник
            season.append(np.sin(2 * np.pi * j * t / k) * (t**i))  # Синусные гармоники
    g_season=np.array(season).T
    G = np.hstack([trend, g_season])
    GtBG=G.T @ B @ G
    XtBG=X.T @ B @ G
    D = np.linalg.solve(GtBG, XtBG)
    proverka=GtBG*D-XtBG
   
    Y= trend @ D[:n + 1] 
    S = g_season @ D[n + 1:]
    E = X - (Y + S)


    t_prognoz=np.arange(dlina,dlina+13)
    trend_prognoz=np.vstack([t_prognoz**i for i in range(n + 1)]).T
    season_prognoz=[]
    for i in range(m+1 ):
        for j in range(1, k // 2 + 1):
            season_prognoz.append(np.cos(2 * np.pi * j * t_prognoz / k) * (t_prognoz**i))  # Косинусные гармоники
        for j in range(1, (k+1) // 2):
            season_prognoz.append(np.sin(2 * np.pi * j * t_prognoz / k) * (t_prognoz**i))  # Синусные гармоники

    season_prognoz = np.array(season_prognoz).T
    Y_test = trend_prognoz @ D[:n + 1]
    S_test = season_prognoz @ D[n + 1:]
    Pro=Y_test+S_test





    return Y,S,E,Pro
def mse_function(lamdaa):
    Y, S, E, Pro = wight_series(X, lamdaa,n,m,k)
    Prognoz = Pro[-12:]
    mse = (1/12*np.sum((X[-12:]-Prognoz)**2))**(1/2)   
    return mse
def moi_vesa(X,weight):






    return Y,S,E
data = load_file()
X=np.array(data)


# Расчет параметров
n_values = [1, 2, 3]      
m_values = [0, 1, 2] 
k = 12 
X_train = X[:-12]
X_val = X[-12:]

best_params=None
best_mse=float('inf')
for n in n_values:
    for m in m_values:
        # Внутри подбираем оптимальное λ
        def mse_for_lambda(lam):
            Y, S, E, Pro = wight_series(X_train, lam, n, m, k)
            mse = np.mean((X_val - Pro[-12:])**2)
            return mse

        try:
            lam_opt = golden_section_search(mse_for_lambda, 0, 1, tol=1e-3)
            final_mse = mse_for_lambda(lam_opt)

            print(f"n={n}, m={m}, λ={lam_opt:.4f}, MSE={final_mse:.4f}")

            if final_mse < best_mse:
                best_mse = final_mse
                best_params = (n, m, lam_opt)
        except Exception as e:
            print(f"Ошибка при n={n}, m={m}: {e}")




dlina=len(data)
t=np.arange(1,dlina+1)
# РАСЧЕТ ВЕСОВ с весом
n=1
m=1

lamdaa_opt = golden_section_search(mse_function,0,1,tol=1e-5)

#Вычисления временного ряда
    

Y,S,E,Pro=wight_series(X,lamdaa_opt,n,m,k)

YS=Y+S
print(np.exp(0.033034))
Prognoz=Pro[1:]
raznica=(Prognoz/data[-12:])

t_t=t[:-12]
t_p=t[-13:]
X_p=X[-13:]
#Нужен прогноз на 12 дней + эти 12 дней должны уже быть в файле 
# Расчет среднеквадратическйо ошибки 
# Q=(1/(12-dlina)*(data)**0.5
seasonal_index = compute_seasonal_index(data[:-12], Y, S, 12)
intervals, frequencies, residual = compute_residual_distribution(data[:-12], Y, S, bins=5)

plt.plot(t_t,YS , color='blue',linewidth=2, linestyle='--',label='Исходные данные')  # Тренд+сезон
plt.plot(t,data , color='green',linewidth=2, label='Исходные данные')  # Исходные данные
plt.plot(t_p,Pro , color='black',linewidth=2,linestyle='--', label='Исходные данные')  # Прогноз
plt.xlabel('Месяцы')
plt.ylabel('Цена р./кг.')
plt.title('Динамика среднемесячных цен на свёклу')
plt.xticks(range(0,len(t)+1,12))
plt.grid(True)
plt.show()
 

plt.figure(figsize=(12, 8))

# Первый график (в верхней части)
plt.subplot(3, 1, 1)  # 2 строки, 1 колонка, 1-й график
plt.plot(t_t, Y, label='sin(x)', color='blue')
plt.xlabel('Месяцы')
plt.xticks(range(0,len(t_t)+1,12))
plt.ylabel('Цена р./кг.')
plt.title('Тренд')

plt.grid()

plt.subplot(3, 1, 2)  # 2 строки, 1 колонка, 2-й график
plt.plot(t_t, S, label='cos(x)', color='red')
plt.xlabel('Месяцы')
plt.xticks(range(0,len(t_t)+1,12))
plt.ylabel('Цена р./кг.')
plt.title('Сезонные колебания m=1')

plt.grid()

plt.subplot(3, 1, 3)  # 2 строки, 1 колонка, 1-й график
plt.plot(t_t, E, label='sin(x)', color='blue')
plt.xlabel('Месяцы')
plt.xticks(range(0,len(t_t)+1,12))
plt.ylabel('Цена р./кг.')
plt.title('Остаточный член')
plt.grid()

# Отображение графика
plt.tight_layout()  # Автоматически подгоняет элементы
plt.show()

plt.figure(figsize=(10, 6))
plt.bar(
    x=(intervals[:-1] + intervals[1:]) / 2, 
    height=frequencies, 
    width=(intervals[1] - intervals[0]) * 0.9, 
    color='skyblue', 
    edgecolor='black',
    label='Частоты остаточного члена'
)
plt.title('Распределение остаточного члена по интервалам', fontsize=14)
plt.xlabel('Интервалы остаточного члена', fontsize=12)
plt.ylabel('Частота', fontsize=12)
plt.grid(alpha=0.7)
plt.legend()
plt.show()

months = ['Янв', 'Фев', 'Мар', 'Апр', 'Май', 'Июн', 'Июл', 'Авг', 'Сен', 'Окт', 'Ноя', 'Дек']
plt.figure(figsize=(10, 6))
plt.bar(months, seasonal_index, color='skyblue', edgecolor='black')
plt.title('Индекс сезонности по месяцам, р\кг.', fontsize=14)
plt.xlabel('Месяц', fontsize=12)
plt.ylabel('Индекс сезонности', fontsize=12)
plt.axhline(1, color='red', linestyle='--', label='Средний уровень')
plt.legend()
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()