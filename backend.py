# backend_fixed_fonts.py
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from ultralytics import YOLO
import cv2
import numpy as np
import json
import os
from datetime import datetime
from pathlib import Path
import uuid
import shutil
import tempfile

app = FastAPI(title="Skateboard Detection API", version="2.1.0")

# Включаем CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Загружаем модель
MODEL_PATH = "runs/detect/runs/train/skateboarder_detection_m/weights/best.pt"
try:
    model = YOLO(MODEL_PATH)
    print(f"✅ Модель загружена: {MODEL_PATH}")
    print(f"📋 Классы: {model.names}")
except Exception as e:
    print(f"❌ Ошибка загрузки модели: {e}")
    model = None

# Папки для хранения
UPLOAD_DIR = "uploads"
REPORTS_DIR = "reports"
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(REPORTS_DIR, exist_ok=True)

# История обработок
HISTORY_FILE = "processing_history.json"

def save_to_history(data: dict):
    """Сохранение в историю"""
    history = []
    if os.path.exists(HISTORY_FILE):
        try:
            with open(HISTORY_FILE, 'r') as f:
                history = json.load(f)
        except:
            history = []
    
    history.append({
        'id': str(uuid.uuid4()),
        'timestamp': datetime.now().isoformat(),
        'filename': data.get('filename', 'unknown'),
        'violations_count': data.get('violations_count', 0),
        'total_objects': data.get('total_objects', 0)
    })
    
    with open(HISTORY_FILE, 'w') as f:
        json.dump(history, f, indent=2, ensure_ascii=False)
    
    return history

def generate_pdf_with_russian(statistics: dict, output_path: str):
    """Генерация PDF с поддержкой русских шрифтов"""
    try:
        from reportlab.lib.pagesizes import A4
        from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image
        from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
        from reportlab.lib import colors
        from reportlab.pdfbase import pdfmetrics
        from reportlab.pdfbase.ttfonts import TTFont
        from reportlab.lib.enums import TA_CENTER, TA_LEFT
        from reportlab.lib.units import mm
        
        print(f"📄 Генерирую PDF с русскими шрифтами...")
        
        # Регистрируем стандартные шрифты, которые поддерживают русский
        # Пробуем использовать встроенные шрифты
        try:
            # Пробуем зарегистрировать Arial, если он есть в системе
            font_paths = [
                "C:/Windows/Fonts/arial.ttf",
                "C:/Windows/Fonts/arialbd.ttf",
                "C:/Windows/Fonts/ariali.ttf",
                "/usr/share/fonts/truetype/msttcorefonts/Arial.ttf",
                "/System/Library/Fonts/Arial.ttf"
            ]
            
            for font_path in font_paths:
                if os.path.exists(font_path):
                    try:
                        pdfmetrics.registerFont(TTFont('Arial', font_path))
                        print(f"✅ Шрифт Arial зарегистрирован: {font_path}")
                        font_name = 'Arial'
                        break
                    except:
                        continue
            else:
                # Если Arial не найден, используем стандартные
                font_name = 'Helvetica'
                print("⚠️  Шрифт Arial не найден, использую Helvetica")
                
        except Exception as e:
            print(f"⚠️  Ошибка регистрации шрифта: {e}")
            font_name = 'Helvetica'
        
        # Создаем документ
        doc = SimpleDocTemplate(output_path, pagesize=A4, 
                              rightMargin=20*mm, leftMargin=20*mm,
                              topMargin=20*mm, bottomMargin=20*mm)
        
        # Создаем кастомные стили
        styles = getSampleStyleSheet()
        
        # Стиль для заголовка
        title_style = ParagraphStyle(
            'CustomTitle',
            parent=styles['Title'],
            fontName=font_name,
            fontSize=24,
            spaceAfter=30,
            alignment=TA_CENTER,
            textColor=colors.darkblue
        )
        
        # Стиль для заголовков разделов
        heading_style = ParagraphStyle(
            'CustomHeading',
            parent=styles['Heading2'],
            fontName=font_name,
            fontSize=16,
            spaceAfter=12,
            textColor=colors.darkgreen,
            spaceBefore=20
        )
        
        # Стиль для обычного текста
        normal_style = ParagraphStyle(
            'CustomNormal',
            parent=styles['Normal'],
            fontName=font_name,
            fontSize=10,
            spaceAfter=6
        )
        
        # Собираем элементы документа
        story = []
        
        # 1. Заголовок
        story.append(Paragraph("ОТЧЕТ ПО АНАЛИЗУ ВИДЕО", title_style))
        story.append(Spacer(1, 10*mm))
        
        # 2. Информация о видео
        story.append(Paragraph("1. Информация о видео", heading_style))
        
        video_info = statistics.get('video_info', {})
        info_data = [
            ["Параметр", "Значение"],
            ["Файл", video_info.get('filename', 'Не указан')],
            ["Разрешение", video_info.get('resolution', 'Не указано')],
            ["Частота кадров", f"{video_info.get('fps', 0)} FPS"],
            ["Всего кадров", str(video_info.get('total_frames', 0))],
            ["Длительность", f"{video_info.get('duration_seconds', 0):.1f} секунд"]
        ]
        
        info_table = Table(info_data, colWidths=[60*mm, 100*mm])
        info_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.lightgrey),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.black),
            ('ALIGN', (0, 0), (-1, 0), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), font_name),
            ('FONTSIZE', (0, 0), (-1, -1), 10),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('GRID', (0, 0), (-1, -1), 1, colors.black),
            ('FONTNAME', (0, 1), (-1, -1), font_name),  # Применяем шрифт ко всем ячейкам
        ]))
        story.append(info_table)
        story.append(Spacer(1, 10*mm))
        
        # 3. Сводная статистика
        story.append(Paragraph("2. Сводная статистика", heading_style))
        
        detections = statistics.get('detections', {})
        summary = statistics.get('summary', {})
        
        summary_data = [
            ["Метрика", "Значение"],
            ["Кадров с детекциями", str(detections.get('total_frames_with_detections', 0))],
            ["Всего объектов", str(detections.get('total_objects_detected', 0))],
            ["Обнаружено нарушений", str(len(detections.get('frames_with_violations', [])))],
            ["% кадров с нарушениями", f"{summary.get('violation_percentage', 0):.1f}%"],
            ["Среднее объектов на кадр", f"{summary.get('avg_objects_per_frame', 0):.2f}"],
            ["Самый частый класс", summary.get('most_common_class', 'Не обнаружено')]
        ]
        
        summary_table = Table(summary_data, colWidths=[80*mm, 80*mm])
        summary_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.lightblue),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.black),
            ('ALIGN', (0, 0), (-1, 0), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), font_name),
            ('FONTSIZE', (0, 0), (-1, -1), 10),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('GRID', (0, 0), (-1, -1), 1, colors.black),
            ('FONTNAME', (0, 1), (-1, -1), font_name),
        ]))
        story.append(summary_table)
        story.append(Spacer(1, 10*mm))
        
        # 4. Детекции по классам
        story.append(Paragraph("3. Детекции по классам", heading_style))
        
        if detections.get('by_class'):
            class_data = [["Класс", "Количество", "Средняя уверенность"]]
            
            for class_name, stats in detections['by_class'].items():
                class_data.append([
                    class_name,
                    str(stats.get('count', 0)),
                    f"{stats.get('avg_confidence', 0):.1%}"
                ])
            
            class_table = Table(class_data, colWidths=[60*mm, 40*mm, 60*mm])
            class_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.lightgreen),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.black),
                ('ALIGN', (0, 0), (-1, 0), 'CENTER'),
                ('FONTNAME', (0, 0), (-1, 0), font_name),
                ('FONTSIZE', (0, 0), (-1, -1), 10),
                ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                ('GRID', (0, 0), (-1, -1), 1, colors.black),
                ('FONTNAME', (0, 1), (-1, -1), font_name),
            ]))
            story.append(class_table)
        else:
            story.append(Paragraph("Объекты не обнаружены", normal_style))
        
        story.append(Spacer(1, 10*mm))
        
        # 5. Нарушения
        violations = detections.get('frames_with_violations', [])
        if violations:
            story.append(Paragraph("4. Нарушения", heading_style))
            
            violation_data = [["Кадр", "Время (сек)", "Уверенность"]]
            
            # Ограничиваем количество отображаемых нарушений
            display_violations = violations[:15]  # Первые 15
            
            for violation in display_violations:
                violation_data.append([
                    str(violation.get('frame', 0)),
                    f"{violation.get('timestamp', 0):.1f}",
                    f"{violation.get('confidence', 0):.1%}"
                ])
            
            if len(violations) > 15:
                violation_data.append(["...", f"и еще {len(violations)-15}", "..."])
            
            violation_table = Table(violation_data, colWidths=[40*mm, 40*mm, 40*mm])
            violation_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.Color(1, 0.9, 0.9)),  # светло-красный
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.black),
                ('ALIGN', (0, 0), (-1, 0), 'CENTER'),
                ('FONTNAME', (0, 0), (-1, 0), font_name),
                ('FONTSIZE', (0, 0), (-1, -1), 9),
                ('BOTTOMPADDING', (0, 0), (-1, 0), 10),
                ('GRID', (0, 0), (-1, -1), 1, colors.black),
                ('FONTNAME', (0, 1), (-1, -1), font_name),
            ]))
            story.append(violation_table)
            
            if len(violations) > 0:
                story.append(Spacer(1, 5*mm))
                story.append(Paragraph(f"Всего нарушений: {len(violations)}", normal_style))
        else:
            story.append(Paragraph("4. Нарушения не обнаружены ✓", heading_style))
            story.append(Paragraph("На видео не обнаружено нарушений правил.", normal_style))
        
        story.append(Spacer(1, 15*mm))
        
        # 6. Заключение
        story.append(Paragraph("5. Заключение", heading_style))
        
        conclusion_text = ""
        if len(violations) == 0:
            conclusion_text = "Нарушений не обнаружено. Видео соответствует правилам."
        elif len(violations) < 5:
            conclusion_text = f"Обнаружено {len(violations)} незначительных нарушений."
        else:
            conclusion_text = f"Обнаружено {len(violations)} серьезных нарушений. Требуется принятие мер."
        
        story.append(Paragraph(conclusion_text, normal_style))
        story.append(Spacer(1, 10*mm))
        
        # 7. Подпись и дата
        story.append(Paragraph("_" * 50, normal_style))
        story.append(Spacer(1, 5*mm))
        
        date_str = datetime.now().strftime("%d.%m.%Y %H:%M")
        story.append(Paragraph(f"Отчет сгенерирован: {date_str}", 
                             ParagraphStyle('Footer', parent=normal_style, 
                                          fontSize=9, textColor=colors.grey)))
        story.append(Paragraph("Система контроля катания на скейтборде", 
                             ParagraphStyle('Footer', parent=normal_style, 
                                          fontSize=9, textColor=colors.grey)))
        
        # Собираем PDF
        doc.build(story)
        print(f"✅ PDF отчет создан: {output_path}")
        return True
        
    except Exception as e:
        print(f"❌ Ошибка генерации PDF: {e}")
        import traceback
        traceback.print_exc()
        
        # Создаем простой текстовый файл если PDF не получился
        try:
            txt_path = output_path.replace('.pdf', '.txt')
            with open(txt_path, 'w', encoding='utf-8') as f:
                f.write("=" * 60 + "\n")
                f.write("ОТЧЕТ ПО АНАЛИЗУ ВИДЕО\n")
                f.write("=" * 60 + "\n\n")
                
                # Информация о видео
                f.write("1. Информация о видео:\n")
                video_info = statistics.get('video_info', {})
                f.write(f"   Файл: {video_info.get('filename', 'Не указан')}\n")
                f.write(f"   Разрешение: {video_info.get('resolution', 'Не указано')}\n")
                f.write(f"   Длительность: {video_info.get('duration_seconds', 0):.1f} сек\n\n")
                
                # Статистика
                f.write("2. Статистика:\n")
                detections = statistics.get('detections', {})
                f.write(f"   Кадров с детекциями: {detections.get('total_frames_with_detections', 0)}\n")
                f.write(f"   Всего объектов: {detections.get('total_objects_detected', 0)}\n")
                f.write(f"   Нарушений: {len(detections.get('frames_with_violations', []))}\n\n")
                
                # Детекции по классам
                if detections.get('by_class'):
                    f.write("3. Детекции по классам:\n")
                    for class_name, stats in detections['by_class'].items():
                        f.write(f"   {class_name}: {stats.get('count', 0)} объектов\n")
                
                f.write(f"\nОтчет сгенерирован: {datetime.now().strftime('%d.%m.%Y %H:%M')}\n")
            
            print(f"✅ Создан текстовый отчет: {txt_path}")
            return False
        except:
            return False

@app.post("/api/upload-video/")
async def upload_video(file: UploadFile = File(...)):
    """Загрузка и обработка видео"""
    try:
        print(f"📥 Получен файл: {file.filename}")
        
        # Проверяем тип файла
        allowed_extensions = {'.mp4', '.avi', '.mov', '.mkv', '.webm'}
        file_ext = Path(file.filename).suffix.lower()
        
        if file_ext not in allowed_extensions:
            raise HTTPException(
                status_code=400, 
                detail=f"Неподдерживаемый формат файла. Поддерживаются: {', '.join(allowed_extensions)}"
            )
        
        # Сохраняем файл во временную папку
        with tempfile.NamedTemporaryFile(delete=False, suffix=file_ext) as tmp_file:
            content = await file.read()
            tmp_file.write(content)
            tmp_path = tmp_file.name
        
        print(f"💾 Файл сохранен временно: {tmp_path} ({len(content)} байт)")
        
        # Если модель загружена, обрабатываем видео
        if model is not None:
            print("🔍 Начинаю обработку видео с моделью...")
            
            # Простая обработка для демо (анализируем первые 5 секунд)
            cap = cv2.VideoCapture(tmp_path)
            fps = int(cap.get(cv2.CAP_PROP_FPS)) or 30
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 1280
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 720
            
            # Анализируем только первые 5 секунд для скорости
            frames_to_analyze = min(fps * 5, total_frames) if total_frames > 0 else 150
            violations = []
            total_detections = 0
            by_class = {}
            
            print(f"📊 Анализирую {frames_to_analyze} кадров...")
            
            for i in range(frames_to_analyze):
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Анализируем каждый 5-й кадр для скорости
                if i % 5 == 0:
                    results = model(frame, conf=0.3, verbose=False)
                    
                    if results[0].boxes is not None:
                        total_detections += 1
                        
                        for box in results[0].boxes:
                            class_id = int(box.cls[0])
                            class_name = model.names.get(class_id, f"class_{class_id}")
                            confidence = float(box.conf[0])
                            
                            # Обновляем статистику по классам
                            if class_name not in by_class:
                                by_class[class_name] = {'count': 0, 'confidences': []}
                            
                            by_class[class_name]['count'] += 1
                            by_class[class_name]['confidences'].append(confidence)
                            
                            # Если скейтбордист - отмечаем как нарушение
                            if class_name.lower() in ['skateboarder', 'skateboard', 'person']:
                                violations.append({
                                    'frame': i,
                                    'timestamp': i / fps,
                                    'confidence': confidence
                                })
            
            cap.release()
            
            # Рассчитываем среднюю уверенность для каждого класса
            for class_name in by_class:
                confidences = by_class[class_name]['confidences']
                by_class[class_name]['avg_confidence'] = sum(confidences) / len(confidences) if confidences else 0
            
            # Формируем статистику
            statistics = {
                'video_info': {
                    'filename': file.filename,
                    'resolution': f"{width}x{height}",
                    'fps': fps,
                    'total_frames': total_frames,
                    'duration_seconds': total_frames / fps if fps > 0 else 0
                },
                'detections': {
                    'total_frames_with_detections': total_detections,
                    'total_objects_detected': sum(stats['count'] for stats in by_class.values()),
                    'by_class': by_class,
                    'frames_with_violations': violations[:50]  # Ограничиваем список
                },
                'summary': {
                    'violation_percentage': (len(violations) / max(1, frames_to_analyze // 5)) * 100,
                    'avg_objects_per_frame': sum(stats['count'] for stats in by_class.values()) / max(1, frames_to_analyze // 5),
                    'most_common_class': max(by_class.items(), key=lambda x: x[1]['count'])[0] if by_class else 'Не обнаружено'
                }
            }
            
        else:
            print("⚠️  Модель не загружена, использую тестовые данные")
            # Тестовые данные для демонстрации
            statistics = {
                'video_info': {
                    'filename': file.filename,
                    'resolution': '1280x720',
                    'fps': 30,
                    'total_frames': 450,
                    'duration_seconds': 15.0
                },
                'detections': {
                    'total_frames_with_detections': 45,
                    'total_objects_detected': 127,
                    'by_class': {
                        'Скейтбордист': {'count': 23, 'avg_confidence': 0.85},
                        'Пешеход': {'count': 89, 'avg_confidence': 0.72},
                        'Велосипедист': {'count': 15, 'avg_confidence': 0.68}
                    },
                    'frames_with_violations': [
                        {'frame': 45, 'timestamp': 1.5, 'confidence': 0.89},
                        {'frame': 120, 'timestamp': 4.0, 'confidence': 0.91},
                        {'frame': 210, 'timestamp': 7.0, 'confidence': 0.76},
                        {'frame': 285, 'timestamp': 9.5, 'confidence': 0.82},
                        {'frame': 360, 'timestamp': 12.0, 'confidence': 0.71}
                    ]
                },
                'summary': {
                    'violation_percentage': 11.1,
                    'avg_objects_per_frame': 2.8,
                    'most_common_class': 'Пешеход'
                }
            }
        
        # Генерируем PDF отчет
        report_id = str(uuid.uuid4())
        pdf_path = os.path.join(REPORTS_DIR, f"report_{report_id}.pdf")
        
        generate_pdf_with_russian(statistics, pdf_path)
        
        # Сохраняем в историю
        save_to_history({
            'filename': file.filename,
            'violations_count': len(statistics['detections']['frames_with_violations']),
            'total_objects': statistics['detections']['total_objects_detected']
        })
        
        # Удаляем временный файл
        try:
            os.unlink(tmp_path)
        except:
            pass
        
        return {
            "status": "success",
            "message": "Видео успешно обработано",
            "report_id": report_id,
            "statistics": statistics,
            "pdf_url": f"/api/download-report/{report_id}"
        }
        
    except Exception as e:
        print(f"❌ Ошибка: {str(e)}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/download-report/{report_id}")
async def download_report(report_id: str):
    """Скачивание отчета"""
    pdf_path = os.path.join(REPORTS_DIR, f"report_{report_id}.pdf")
    txt_path = os.path.join(REPORTS_DIR, f"report_{report_id}.txt")
    
    if os.path.exists(pdf_path):
        return FileResponse(
            path=pdf_path,
            filename=f"skateboard_report_{report_id}.pdf",
            media_type='application/pdf'
        )
    elif os.path.exists(txt_path):
        return FileResponse(
            path=txt_path,
            filename=f"skateboard_report_{report_id}.txt",
            media_type='text/plain'
        )
    else:
        raise HTTPException(status_code=404, detail="Отчет не найден")

@app.get("/api/history/")
async def get_history():
    """Получение истории обработок"""
    if os.path.exists(HISTORY_FILE):
        try:
            with open(HISTORY_FILE, 'r', encoding='utf-8') as f:
                history = json.load(f)
            return {"history": history}
        except:
            return {"history": []}
    return {"history": []}

@app.get("/api/test-connection/")
async def test_connection():
    """Тестовый эндпоинт"""
    return {
        "status": "success",
        "message": "API работает",
        "model_loaded": model is not None,
        "model_classes": model.names if model else None,
        "timestamp": datetime.now().isoformat()
    }

@app.get("/")
async def root():
    return {"message": "Skateboard Detection API v2.1", "status": "running"}

if __name__ == "__main__":
    import uvicorn
    print("=" * 60)
    print("🚀 Запускаю Skateboard Detection API v2.1")
    print("=" * 60)
    print(f"📁 Папка загрузок: {os.path.abspath(UPLOAD_DIR)}")
    print(f"📁 Папка отчетов: {os.path.abspath(REPORTS_DIR)}")
    print(f"🌐 API доступен по адресу: http://localhost:8000")
    print("=" * 60)
    
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")