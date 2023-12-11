import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from scipy.stats import ttest_ind
import streamlit as st


uploaded_file = st.file_uploader("Загрузите свой CSV файл", type=['csv'])

if uploaded_file is not None:
    
    data = pd.read_csv(uploaded_file, delimiter=',', encoding='windows-1251', quotechar='"')

    data.columns=[it.replace('"','') for it in data.columns]
    data['work_days'] = data['Количество больничных дней,Возраст,Пол'].str.split(',').str[0].astype(int)
    data['age'] = data['Количество больничных дней,Возраст,Пол'].str.split(',').str[1].astype(int)
    data['gender'] = data['Количество больничных дней,Возраст,Пол'].str.split(',').str[2].str.replace('"','')
    data.drop(columns=['Количество больничных дней,Возраст,Пол'], inplace=True)
    
    st.write("Первые строки данных:")
    st.write(data.head())
    
    
    st.write('Руководство компании обратило внимание на то, что сотрудники старше 35 лет болеют чаще, чем более молодые сотрудники.')
    

    
    st.write('Для начала необходимо проверить следующие гипотезы:')
    st.write('1) Мужчины пропускают в течение года более X рабочих дней (work_days) по болезни значимо чаще женщин.')
    st.write('2) Работники старше Y лет (age) пропускают в течение года более X рабочих дней (work_days) по болезни значимо чаще своих более молодых коллег')
    
    st.write('Зададим два параметра:')
    
    days = st.slider("X(количество дней)", min_value=1, max_value=10)
    age = st.slider("Y(пороговый возраст)", min_value=18, max_value=60)
    
    st.header('Гипотеза 1')
    st.write(f'Мужчины пропускают в течение года более {days} рабочих дней по болезни значимо чаще женщин.')
    
    st.write('Для начала посмотрим распределение пропусков по болезни поближе.')
    
    
    plt.figure(figsize=(8, 6))
    plt.hist(data['work_days'], bins=20, color='skyblue', edgecolor='black')
    plt.xlabel('Количество больничных дней')
    plt.ylabel('Частота')
    plt.title('Распределение пропусков по болезни')
    plt.grid(True)
    st.pyplot(plt)
    
    
    st.write("и распределение по полу")
    
    grouped_data = data[data['work_days']>days].groupby('gender')['work_days'].sum()
    
    plt.figure(figsize=(6, 5))
    grouped_data.plot(kind='bar', color=['blue', 'pink'])
    plt.title('Распределение пропусков по болезни по полу')
    plt.xlabel('Пол')
    plt.ylabel('Общее количество больничных дней')
    plt.xticks(rotation=0)
    for index, value in enumerate(grouped_data):
        plt.text(index, value + 5, f'{value}', ha='center', va='bottom')
    
    plt.legend()
    plt.tight_layout()
    st.pyplot(plt)
    
    st.write("Т.к. распределение немного скошено влево, лучше использую бутстрап при применении t-test далее")
    
    def bootstrap_ttest(data1, data2, n_iter=1000):
        t_statistic = np.zeros(n_iter)
        for i in range(n_iter):
            combined = np.concatenate((data1, data2))
            np.random.shuffle(combined)
            perm_sample1 = combined[:len(data1)]
            perm_sample2 = combined[len(data1):]
            t_statistic[i] = abs(ttest_ind(perm_sample1, perm_sample2, equal_var=False).statistic)
        return t_statistic
    
    st.write("Теперь сделаем статистическую проверку")
    
    q1 = data[data['work_days'] > days]
    male_data = q1[q1['gender'] == 'М']
    female_data = q1[q1['gender'] == 'Ж']
    
    boot_t_statistic = bootstrap_ttest(male_data.work_days, female_data.work_days)
    p_value = (np.sum(boot_t_statistic > abs(ttest_ind(male_data.work_days, female_data.work_days, equal_var=False).statistic)) + 1) / (len(boot_t_statistic) + 1)
    
    st.write("Результаты:")
    st.write(f"p-value: {p_value}\n")
    
    if p_value < 0.05:
        st.subheader("Вывод - Есть статистически значимые различия между группами.")
    else:
        st.subheader("Вывод - Нет статистически значимых различий между группами.")
    
    st.write("То есть, хотя количество пропущенных рабочих дней могут отличаться между мужчинами и женщинами, эта разница не считается статистически значимой на уровне 0.05.")
    
    st.header("Гипотеза 2")
    st.write(f"Работники старше {age} лет пропускают в течение года более {days} рабочих дней по болезни значимо чаще своих более молодых коллег")
    
    
    q2 = data[data['work_days'] > 2]
    # деление данных на две группы
    older_than_age = q2[q2['age'] > age]['work_days']
    k_older_than_age = older_than_age.sum()
    younger_than_age = q2[q2['age'] <= age]['work_days']
    k_younger_than_age = younger_than_age.sum()
    
    st.write("посмотрю какое распределение")
    
    plt.figure(figsize=(8, 6))
    plt.hist(data['age'], color='skyblue', edgecolor='black')
    plt.xlabel('Возраст')
    plt.ylabel('Количество')
    plt.title('Распределение по возрастам')
    plt.grid(True)
    st.pyplot(plt)
    
    plt.figure(figsize=(6, 6))
    plt.bar([f'Старше {age} лет', f'Моложе {age} лет'], [k_older_than_age, k_younger_than_age], color=['orange', 'green'])
    plt.title('Общее количество пропущенных рабочих дней\n по возрастным группам')
    plt.ylabel('Общее количество пропущенных рабочих дней')
    plt.xlabel('Возрастная группа')
    
    plt.text(0, k_older_than_age + 5, f'{k_older_than_age}', ha='center', va='bottom', color='black')
    plt.text(1, k_younger_than_age + 5, f'{k_younger_than_age}', ha='center', va='bottom', color='black')
    
    plt.tight_layout()
    st.pyplot(plt)
    
    st.write("распределение близко к нормальному. Использую t-test как есть")
    
    t_stat, p_value = ttest_ind(older_than_age, younger_than_age)
    st.write(f"Значение t-статистики: {t_stat}")
    st.write("Результаты статистической проверки гипотезы:")
    st.write(f"p-value: {p_value}\n")
    
    if p_value < 0.05:
        st.subheader("Вывод - Есть статистически значимые различия между группами.")
    else:
        st.subheader("Вывод - Нет статистически значимых различий между группами.")
    
    st.write(f"разница не считается статистически значимой на уровне 0.05, поэтому не можем сказать, что работники старше {age} лет более чаще болеют, чем молодые коллеги")



