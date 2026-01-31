# frontend_fixed.py
import streamlit as st
import requests
import pandas as pd
from datetime import datetime
import time
import json

# Конфигурация
BACKEND_URL = "http://localhost:8000"
st.set_page_config(
    page_title="Контроль катания на скейтборде",
    page_icon="🛹",
    layout="wide"
)

st.title("🛹 Контроль катания на скейтборде")
st.markdown("---")

# Проверка подключения к бекенду
try:
    response = requests.get(f"{BACKEND_URL}/api/test-connection/", timeout=5)
    if response.status_code == 200:
        st.success("✅ Подключено к бекенду")
        data = response.json()
        if data.get('model_loaded'):
            st.info(f"🤖 Модель загружена. Классы: {data.get('model_classes')}")
        else:
            st.warning("⚠️ Модель не загружена")
    else:
        st.error("❌ Бекенд недоступен")
except:
    st.error("❌ Не удалось подключиться к бекенду")
    st.info("Запустите бекенд: `python backend.py`")

st.markdown("---")

# Загрузка видео
st.header("📤 Загрузка видео")
uploaded_file = st.file_uploader(
    "Выберите видео файл для анализа",
    type=['mp4', 'avi', 'mov', 'mkv'],
    help="Максимальный размер: 100 MB"
)

if uploaded_file:
    st.info(f"📁 Выбран файл: **{uploaded_file.name}** ({uploaded_file.size / 1024 / 1024:.1f} MB)")
    
    if st.button("🚀 Начать анализ видео", type="primary"):
        # Показываем прогресс
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        try:
            # Загружаем файл
            status_text.text("📤 Загружаю видео на сервер...")
            
            files = {"file": (uploaded_file.name, uploaded_file, uploaded_file.type)}
            response = requests.post(f"{BACKEND_URL}/api/upload-video/", files=files)
            
            progress_bar.progress(30)
            status_text.text("🔍 Анализирую видео...")
            
            if response.status_code == 200:
                result = response.json()
                progress_bar.progress(70)
                status_text.text("📊 Формирую отчет...")
                
                # Показываем результаты
                st.success("✅ Видео успешно обработано!")
                
                # Статистика
                stats = result['statistics']
                
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    duration = stats['video_info']['duration_seconds']
                    st.metric("Длительность", f"{duration} сек")
                
                with col2:
                    res = stats['video_info']['resolution']
                    st.metric("Разрешение", res)
                
                with col3:
                    violations = len(stats['detections']['frames_with_violations'])
                    st.metric("Нарушения", violations)
                
                with col4:
                    st.metric("Объектов", stats['detections']['total_objects_detected'])
                
                # Таблица с детекциями
                if stats['detections']['by_class']:
                    st.subheader("🎯 Детекции по классам")
                    
                    detection_data = []
                    for class_name, class_stats in stats['detections']['by_class'].items():
                        detection_data.append({
                            'Класс': class_name,
                            'Количество': int(class_stats['count']),  # Явно указываем int
                            'Средняя уверенность': float(class_stats['avg_confidence'])  # Явно указываем float
                        })
                    
                    df = pd.DataFrame(detection_data)
                    # Форматируем проценты
                    if 'Средняя уверенность' in df.columns:
                        df['Средняя уверенность'] = df['Средняя уверенность'].apply(lambda x: f"{x:.1%}")
                    
                    # Исправлено: используем width вместо use_container_width
                    st.dataframe(df, width='stretch')
                
                # Нарушения
                if stats['detections']['frames_with_violations']:
                    st.subheader("⚠️ Нарушения")
                    violations_df = pd.DataFrame(stats['detections']['frames_with_violations'])
                    
                    # Конвертируем типы данных
                    if 'confidence' in violations_df.columns:
                        violations_df['confidence'] = violations_df['confidence'].apply(lambda x: f"{x:.1%}")
                    if 'timestamp' in violations_df.columns:
                        violations_df['timestamp'] = violations_df['timestamp'].apply(lambda x: f"{x:.1f} сек")
                    
                    st.dataframe(violations_df, width='stretch')
                
                # Кнопка скачивания PDF
                st.markdown("---")
                st.subheader("📄 PDF отчет")
                
                report_id = result['report_id']
                pdf_url = f"{BACKEND_URL}/api/download-report/{report_id}"
                
                st.markdown(f"[📥 Скачать полный PDF отчет]({pdf_url})")
                
                # Альтернативный способ скачивания
                if st.button("💾 Сохранить отчет локально", type="secondary"):
                    try:
                        pdf_response = requests.get(pdf_url)
                        if pdf_response.status_code == 200:
                            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                            filename = f"skateboard_report_{timestamp}.pdf"
                            
                            with open(filename, "wb") as f:
                                f.write(pdf_response.content)
                            
                            st.success(f"✅ Отчет сохранен как: {filename}")
                            
                            # Предлагаем скачать
                            with open(filename, "rb") as f:
                                st.download_button(
                                    label="📥 Скачать файл",
                                    data=f,
                                    file_name=filename,
                                    mime="application/pdf"
                                )
                        else:
                            st.error("Не удалось скачать отчет")
                    except Exception as e:
                        st.error(f"Ошибка скачивания: {e}")
                
                progress_bar.progress(100)
                status_text.text("✅ Готово!")
                
                # Показать сырые данные
                with st.expander("📊 Показать полные данные"):
                    st.json(result)
                
            else:
                st.error(f"❌ Ошибка обработки: {response.text}")
                
        except Exception as e:
            st.error(f"❌ Ошибка: {str(e)}")
            import traceback
            st.code(traceback.format_exc())

# История обработок
st.markdown("---")
if st.button("📜 Показать историю обработок", type="secondary"):
    try:
        response = requests.get(f"{BACKEND_URL}/api/history/")
        if response.status_code == 200:
            history = response.json().get('history', [])
            
            if history:
                st.subheader("История обработок")
                
                # Конвертируем в DataFrame
                df = pd.DataFrame(history)
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                df['Дата'] = df['timestamp'].dt.strftime('%Y-%m-%d %H:%M')
                
                # Конвертируем числовые колонки
                numeric_cols = ['violations_count', 'total_objects']
                for col in numeric_cols:
                    if col in df.columns:
                        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0).astype(int)
                
                # Показываем таблицу
                st.dataframe(
                    df[['Дата', 'filename', 'violations_count', 'total_objects']],
                    width='stretch'  # Исправлено
                )
                
                # Простая статистика
                if len(df) > 0:
                    st.write(f"**Всего обработок:** {len(df)}")
                    st.write(f"**Всего нарушений:** {df['violations_count'].sum()}")
                    st.write(f"**Среднее нарушений на видео:** {df['violations_count'].mean():.1f}")
            else:
                st.info("История обработок пуста")
    except Exception as e:
        st.error(f"Не удалось загрузить историю: {e}")

# Информация о системе
with st.expander("ℹ️ Информация о системе"):
    st.write(f"**Бекенд:** {BACKEND_URL}")
    st.write(f"**Текущее время:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    try:
        test_resp = requests.get(f"{BACKEND_URL}/api/test-connection/")
        if test_resp.status_code == 200:
            data = test_resp.json()
            st.write("**Статус API:** ✅ Работает")
            st.write(f"**Модель загружена:** {'✅ Да' if data.get('model_loaded') else '❌ Нет'}")
            if data.get('model_classes'):
                st.write(f"**Классы модели:** {data.get('model_classes')}")
    except:
        st.write("**Статус API:** ❌ Недоступен")

# Простой тест загрузки файла
st.markdown("---")
st.header("🧪 Тест системы")

col1, col2 = st.columns(2)

with col1:
    if st.button("🔄 Проверить подключение"):
        try:
            response = requests.get(f"{BACKEND_URL}/", timeout=3)
            if response.status_code == 200:
                st.success(f"✅ API работает: {response.json()}")
            else:
                st.error(f"❌ API вернул код: {response.status_code}")
        except Exception as e:
            st.error(f"❌ Ошибка подключения: {e}")

with col2:
    if st.button("🗑️ Очистить историю"):
        try:
            # Создаем простой эндпоинт для очистки
            if os.path.exists("processing_history.json"):
                os.remove("processing_history.json")
                st.success("✅ История очищена")
            else:
                st.info("Файл истории не найден")
        except Exception as e:
            st.error(f"❌ Ошибка: {e}")
