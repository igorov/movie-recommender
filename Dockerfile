# Imagen base
FROM python:3.9-slim-buster

# Directorio de trabajo
WORKDIR /app

# Copiar archivos necesarios
COPY requirements.txt .
COPY . .

# Instalar dependencias
RUN pip install --no-cache-dir -r requirements.txt

# Exponer puerto
EXPOSE 8080

# Comando para iniciar la aplicación
CMD ["python", "src/main.py"]