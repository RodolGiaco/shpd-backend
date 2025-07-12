#!/bin/bash

# Script para generar PDF de la documentación del backend
# Requiere: pandoc, pdflatex (texlive-full)

echo "Generando PDF de la documentación del backend..."

# Verificar que pandoc esté instalado
if ! command -v pandoc &> /dev/null; then
    echo "Error: Pandoc no está instalado. Por favor instálalo primero."
    echo "Ubuntu/Debian: sudo apt-get install pandoc texlive-full"
    echo "MacOS: brew install pandoc mactex"
    exit 1
fi

# Generar PDF con metadatos
pandoc documentacion_backend_completa.md \
    --metadata-file=metadata.yaml \
    --pdf-engine=pdflatex \
    --highlight-style=tango \
    --listings \
    -o documentacion_backend_tesis.pdf

# Verificar si se generó correctamente
if [ -f "documentacion_backend_tesis.pdf" ]; then
    echo "✅ PDF generado exitosamente: documentacion_backend_tesis.pdf"
    echo "Tamaño del archivo: $(ls -lh documentacion_backend_tesis.pdf | awk '{print $5}')"
else
    echo "❌ Error al generar el PDF"
    exit 1
fi

# Opcional: Abrir el PDF generado (descomentar según el sistema)
# Linux:
# xdg-open documentacion_backend_tesis.pdf

# MacOS:
# open documentacion_backend_tesis.pdf

# Windows (WSL):
# cmd.exe /c start documentacion_backend_tesis.pdf