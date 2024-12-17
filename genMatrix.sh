#!/bin/bash

# Crear la carpeta matrix si no existe
mkdir -p matrix

# Lista de URLs
urls=(
    "https://suitesparse-collection-website.herokuapp.com/MM/GHS_psdef/bmwcra_1.tar.gz"
    "https://suitesparse-collection-website.herokuapp.com/MM/GHS_psdef/apache2.tar.gz"
    "https://suitesparse-collection-website.herokuapp.com/MM/UTEP/Dubcova2.tar.gz"
    "https://suitesparse-collection-website.herokuapp.com/MM/UTEP/Dubcova3.tar.gz"
    "https://suitesparse-collection-website.herokuapp.com/MM/Botonakis/FEM_3D_thermal1.tar.gz"
    "https://suitesparse-collection-website.herokuapp.com/MM/AMD/G3_circuit.tar.gz"
    "https://suitesparse-collection-website.herokuapp.com/MM/Mulvey/finan512.tar.gz"
    "https://suitesparse-collection-website.herokuapp.com/MM/Oberwolfach/boneS01.tar.gz"
    "https://suitesparse-collection-website.herokuapp.com/MM/GHS_indef/c-70.tar.gz"
    "https://suitesparse-collection-website.herokuapp.com/MM/Williams/consph.tar.gz"
    "https://suitesparse-collection-website.herokuapp.com/MM/Boeing/ct20stif.tar.gz"
    "https://suitesparse-collection-website.herokuapp.com/MM/McRae/ecology2.tar.gz"
    "https://suitesparse-collection-website.herokuapp.com/MM/TKK/engine.tar.gz"
    "https://suitesparse-collection-website.herokuapp.com/MM/Oberwolfach/filter3D.tar.gz"
    "https://suitesparse-collection-website.herokuapp.com/MM/Wissgott/parabolic_fem.tar.gz"
    "https://suitesparse-collection-website.herokuapp.com/MM/Boeing/pwtk.tar.gz"
    "https://suitesparse-collection-website.herokuapp.com/MM/MaxPlanck/shallow_water1.tar.gz"
    "https://suitesparse-collection-website.herokuapp.com/MM/Norris/torso3.tar.gz"
    "https://suitesparse-collection-website.herokuapp.com/MM/Simon/venkat50.tar.gz"
)

# Descargar cada archivo
for url in "${urls[@]}"; do
    # Extraer el nombre del archivo
    filename=$(basename "$url")
    # Descargar el archivo en la carpeta matrix
    echo "Descargando $filename..."
    curl -L -o "matrix/$filename" "$url"

    # Descomprimir el archivo tar.gz
    echo "Descomprimiendo $filename..."
    tar -xzf "matrix/$filename" -C matrix

    # Eliminar el archivo .tar.gz
    rm "matrix/$filename"

    # Buscar la carpeta descomprimida y mover el archivo .mtx
    extracted_folder=$(basename "$filename" .tar.gz)
    mtx_file=$(find "matrix/$extracted_folder" -name "*.mtx")

    if [[ -n "$mtx_file" ]]; then
        # Mover el archivo .mtx a la carpeta matrix
        mv "$mtx_file" matrix/
        # Eliminar la carpeta vacÃa
        #rm -r "matrix/$extracted_folder"
    fi
done

echo "Todos los archivos han sido descargados, descomprimidos y organizados correctamente."
