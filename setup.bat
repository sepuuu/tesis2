@echo off
echo Configurando el entorno para el proyecto...

:: Crear el entorno virtual con Python 3.11
echo Creando entorno virtual con Python 3.11...
py -m venv env

:: Activar el entorno virtual
echo Activando el entorno virtual...
call env\Scripts\activate

:: Instalar dependencias
echo Instalando dependencias...
pip install -r requirements.txt

:: Configuración completada
echo Configuración completada. Para activar el entorno nuevamente, usa "env\Scripts\activate".