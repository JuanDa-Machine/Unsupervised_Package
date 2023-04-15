import dash
from dash.dependencies import Input, Output, State
from dash import dcc, html
import numpy as np
import pandas as pd
import io
from PIL import Image
import plotly.express as px
import matplotlib.pyplot as plt
import base64

from packages.main import pca
from packages.main import svd

# Definir la aplicación Dash
app = dash.Dash(__name__)

# Definir el diseño de la aplicación
app.layout = html.Div([
    html.H1("Aplicación PCA Y SVD"),
    dcc.Upload(
        id='upload-image',
        children=html.Div([
            'Arrastre o seleccione la imagen'
        ]),
        style={
            'width': '50%',
            'height': '60px',
            'lineHeight': '60px',
            'borderWidth': '1px',
            'borderStyle': 'dashed',
            'borderRadius': '5px',
            'textAlign': 'center',
            'margin': '10px'
        },
        multiple=False
    ),
    #html.Div(id='output-image-upload'),
    html.Div(id='output-pca'), 
    html.Div(id='output-svd')
])

# Definir la función para cargar y preprocesar la imagen
def load_image(image_string):
    try:
        # Convertir la cadena de caracteres a bytes
        image_bytes = base64.b64decode(image_string.split(',')[1])
        #image_bytes = image_string.split(',')[1]
        img = Image.open(io.BytesIO(image_bytes))
        size= (256, 256)
        img= img.resize(size)
        img= img.convert('L')
        img = np.array(img)
        img_df = img.reshape(-1, img.shape[-1])
        #img_df = pd.DataFrame(img)
        return img_df
    except:
        return None
 


# Definir la función para aplicar el PCA a la imagen
def apply_pca(image_df):
    #image_df = image_df.reshape(image_df.shape)
    #image_df = image_df.astype('float32') / 255
    pcaR = pca.PCA(n_components= 50)
    pcaR.fit(image_df)
    pca_img = pcaR.transform(image_df)
    #pca_img = pca_img.reshape(*image_df.shape)
    pca_img = pca_img.astype('uint8')
    return pca_img
    
def apply_svd(image_df):
    svdR= svd.svd(image_df)
    return svdR
    
    

# Definir la función para codificar la imagen en base64 y mostrarla en la aplicación
def encode_image(image):
    buffered = io.BytesIO()
    image.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode()
    return f"data:image/jpeg;base64,{img_str}"

# Definir las funciones de devolución de llamada para la carga de la imagen y la aplicación del PCA
@app.callback(#Output('output-image-upload', 'children'),
              Output('output-pca', 'children'),
              Output('output-svd', 'children'),
              Input('upload-image', 'contents'))
def update_output(image_string):
    if image_string is not None:
        # Cargar y preprocesar la imagen
        img_df = load_image(image_string)

        # Aplicar el PCA a la imagen
        pca_img = apply_pca(img_df)
        
        svd_img= apply_svd(img_df)

        # Codificar la imagen resultante en base64 y mostrarla en la aplicación
        pca_img_encoded = encode_image(Image.fromarray(pca_img))
        svd_img_encoded = encode_image(Image.fromarray(svd_img).convert('RGB'))
        #pca_fig = px.imshow(pca_img)
        #pca_fig.update_layout(
        #    title="Imagen con PCA aplicado"
        #)

        return html.Div([
            html.Div([
                html.H5("Imagen original:"),
                html.Img(src=image_string, width=400)
            ]),
            html.Div([
                html.H5("Imagen con PCA aplicado:"),
                html.Img(src=pca_img_encoded, width=400, height= 400)
            ]), 
            html.Div([
                html.H5("Imagen con SVD aplicado:"),
                html.Img(src=svd_img_encoded, width=400, height= 400)
            ])
        ]), None

    return html.Div([
        'Arrastre o seleccione la imagen'
    ]), None

# Iniciar la aplicación Dash

if __name__ == '__main__':
    app.run_server(debug=True)
