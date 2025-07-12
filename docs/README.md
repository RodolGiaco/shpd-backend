# Documentación del Backend - Sistema de Detección de Posturas

Este directorio contiene la documentación técnica completa del backend del Sistema de Detección de Posturas, desarrollada para una tesis de ingeniería.

## Contenido

- `documentacion_backend_completa.md`: Documento principal con toda la documentación técnica
- `metadata.yaml`: Metadatos para la generación del PDF con formato profesional
- `generar_pdf.sh`: Script para generar el PDF usando Pandoc
- `README.md`: Este archivo

## Requisitos para Generar el PDF

### Ubuntu/Debian
```bash
sudo apt-get update
sudo apt-get install pandoc texlive-full
```

### MacOS
```bash
brew install pandoc
brew install --cask mactex
```

### Windows
- Instalar [Pandoc](https://pandoc.org/installing.html)
- Instalar [MiKTeX](https://miktex.org/download)

## Generar el PDF

1. Navegar al directorio de documentación:
```bash
cd docs/
```

2. Ejecutar el script de generación:
```bash
./generar_pdf.sh
```

3. El PDF se generará como `documentacion_backend_tesis.pdf`

## Generación Manual

Si prefieres generar el PDF manualmente:

```bash
pandoc documentacion_backend_completa.md \
    --metadata-file=metadata.yaml \
    --pdf-engine=pdflatex \
    --highlight-style=tango \
    --listings \
    -o documentacion_backend_tesis.pdf
```

## Estructura de la Documentación

1. **Introducción**: Visión general del proyecto
2. **Arquitectura General**: Diseño del sistema
3. **Tecnologías**: Stack tecnológico utilizado
4. **Módulo Principal**: Análisis detallado de main.py
5. **Monitor de Posturas**: Sistema de detección con MediaPipe
6. **Capa de Datos**: Modelos y esquemas de base de datos
7. **API REST**: Documentación de todos los endpoints
8. **WebSockets**: Sistema de comunicación en tiempo real
9. **Integración OpenAI**: Análisis con inteligencia artificial
10. **Redis**: Sistema de caché y estado temporal
11. **Despliegue**: Docker y Kubernetes
12. **Casos de Uso**: Flujos principales del sistema
13. **Análisis**: Fortalezas y mejoras propuestas
14. **Conclusiones**: Logros e impacto del proyecto

## Personalización

Para personalizar los metadatos del documento (autor, título, etc.), edita el archivo `metadata.yaml`.

## Formatos Alternativos

### Generar HTML
```bash
pandoc documentacion_backend_completa.md \
    --metadata-file=metadata.yaml \
    --standalone \
    --toc \
    -o documentacion_backend_tesis.html
```

### Generar DOCX
```bash
pandoc documentacion_backend_completa.md \
    --metadata-file=metadata.yaml \
    --reference-doc=reference.docx \
    -o documentacion_backend_tesis.docx
```

## Notas

- La documentación está escrita en español técnico formal
- Incluye explicaciones línea por línea del código cuando es necesario
- Compatible con los requisitos de documentación para tesis de ingeniería
- El formato sigue las mejores prácticas para documentación técnica

## Soporte

Si encuentras algún problema al generar el PDF, verifica:
1. Que Pandoc esté correctamente instalado: `pandoc --version`
2. Que LaTeX esté disponible: `pdflatex --version`
3. Que todos los archivos necesarios estén en el directorio

---

*Documentación generada para proyecto de tesis de ingeniería*