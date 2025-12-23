#!/bin/bash
set -e

echo "=== FFT Correlator Reports Viewer ==="
echo

# Показать последний Markdown отчет профилирования
echo "📊 Последний отчет профилирования:"
LATEST_MD=$(find build/Report -name "*.md" -type f -printf '%T@ %p\n' 2>/dev/null | sort -n | tail -1 | cut -d' ' -f2-)
if [ -n "$LATEST_MD" ]; then
    echo "Файл: $LATEST_MD"
    echo "----------------------------------------"
    head -20 "$LATEST_MD"
    echo "----------------------------------------"
    echo "(Полный отчет: cat \"$LATEST_MD\")"
else
    echo "Markdown отчеты не найдены"
fi

echo
echo "📋 Последние JSON отчеты валидации:"
echo

# Показать последние JSON файлы
find Report/Validation -name "*.json" -type f -printf '%T@ %p\n' 2>/dev/null | sort -n | tail -5 | while read timestamp filepath; do
    filename=$(basename "$filepath")
    echo "📄 $filename"
    echo "   Путь: $filepath"
    echo "   Время: $(date -d "@$timestamp" '+%Y-%m-%d %H:%M:%S')"
    echo
done

echo "💡 Для просмотра полного JSON отчета используйте:"
echo "   jq . Report/Validation/[filename].json | less"
echo
echo "📈 Для просмотра всех отчетов:"
echo "   ls -la Report/"
echo "   ls -la Report/Validation/"
echo "   ls -la Report/JSON/"
