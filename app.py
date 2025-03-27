# =============================================================================
# Importação de Bibliotecas 
# =============================================================================

import streamlit as st
import pandas as pd
import numpy as np
import requests
import folium
import geobr
import matplotlib.pyplot as plt
import seaborn as sns
from streamlit_folium import st_folium
from folium.plugins import Draw, HeatMap
from shapely.geometry import Point, Polygon
from shapely.ops import unary_union
import time
import ee
import json
import os
import logging
import yaml
from scipy.stats import gaussian_kde
import concurrent.futures
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tools.tools import add_constant
from google.oauth2 import service_account
from ee import oauth
from typing import Dict, List, Tuple, Optional, Union, Any
import rtree  # Para indexação espacial
from functools import wraps
from contextlib import nullcontext


# =============================================================================
# Configuração de Logging
# =============================================================================

# Configuração do sistema de logging
def configurar_logging():
    """
    Configura o sistema de logging da aplicação.
    
    Define dois handlers: um para o console e outro para um arquivo,
    com formatação personalizada para facilitar o debugging.
    """
    # Certifique-se de que o diretório de logs existe
    if not os.path.exists('logs'):
        os.makedirs('logs')
    
    # Configuração do logger
    logger = logging.getLogger('taipa')
    logger.setLevel(logging.INFO)
    
    # Limpa handlers existentes para evitar duplicação
    if logger.handlers:
        logger.handlers.clear()
    
    # Handler para console
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_format = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(console_format)
    
    # Handler para arquivo
    file_handler = logging.FileHandler('logs/taipa.log')
    file_handler.setLevel(logging.DEBUG)
    file_format = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(file_format)
    
    # Adiciona os handlers ao logger
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)
    
    return logger

# Inicializa o logger
logger = configurar_logging()


# =============================================================================
# Carregamento de Configuração
# =============================================================================

def carregar_configuracao():
    """
    Carrega a configuração de um arquivo YAML.
    
    Tenta carregar a configuração de 'config.yaml'. Se o arquivo não existir,
    cria uma configuração padrão e a salva no arquivo.
    
    Returns:
        dict: Dicionário com as configurações da aplicação.
    """
    try:
        # Verifica se o arquivo de configuração existe
        if not os.path.exists('config.yaml'):
            # Configuração padrão
            config = {
                'api': {
                    'gbif': {
                        'base_url': 'https://api.gbif.org/v1/occurrence/search',
                        'limite_padrao': 100,
                        'limite_maximo': 10000
                    }
                },
                'pseudoausencias': {
                    'padrao': 100,
                    'minimo': 50,
                    'maximo': 500,
                    'buffer_padrao_km': 50,
                    'max_tentativas_por_ponto': 1000
                },
                'vis_params': {
                    1:  {'min': 10,  'max': 30,   'palette': ['blue', 'cyan', 'green', 'yellow', 'red']},
                    2:  {'min': 0,   'max': 20,   'palette': ['white', 'blue']},
                    3:  {'min': 0,   'max': 100,  'palette': ['yellow', 'orange', 'red']},
                    4:  {'min': 0,   'max': 500,  'palette': ['white', 'blue']},
                    5:  {'min': 10,  'max': 50,   'palette': ['white', 'red']},
                    6:  {'min': 0,   'max': 30,   'palette': ['white', 'blue']},
                    7:  {'min': 0,   'max': 50,   'palette': ['white', 'green']},
                    8:  {'min': 5,   'max': 40,   'palette': ['yellow', 'red']},
                    9:  {'min': 0,   'max': 30,   'palette': ['white', 'yellow', 'blue']},
                    10: {'min': 10,  'max': 30,   'palette': ['white', 'red']},
                    11: {'min': 0,   'max': 30,   'palette': ['blue', 'green', 'yellow', 'red']},
                    12: {'min': 200, 'max': 3000, 'palette': ['red', 'yellow', 'blue']},
                    13: {'min': 0,   'max': 300,  'palette': ['red', 'yellow', 'blue']},
                    14: {'min': 0,   'max': 200,  'palette': ['red', 'yellow', 'blue']},
                    15: {'min': 0,   'max': 100,  'palette': ['white', 'orange', 'red']},
                    16: {'min': 0,   'max': 1000, 'palette': ['white', 'lightblue', 'blue']},
                    17: {'min': 0,   'max': 300,  'palette': ['red', 'yellow','blue']},
                    18: {'min': 0,   'max': 1000, 'palette': ['white', 'lightblue', 'blue']},
                    19: {'min': 0,   'max': 800,  'palette': ['white', 'lightblue', 'blue']}
                },
                'descricoes_bio': {
                    1: "Temperatura Média Anual",
                    2: "Amplitude Diurna Média (Média mensal de (temp. máxima - temp. mínima))",
                    3: "Isotermalidade (BIO2/BIO7) (×100)",
                    4: "Sazonalidade da Temperatura (desvio padrão ×100)",
                    5: "Temperatura Máxima do Mês Mais Quente",
                    6: "Temperatura Mínima do Mês Mais Frio",
                    7: "Amplitude Anual da Temperatura (BIO5 - BIO6)",
                    8: "Temperatura Média do Trimestre Mais Úmido",
                    9: "Temperatura Média do Trimestre Mais Seco",
                    10: "Temperatura Média do Trimestre Mais Quente",
                    11: "Temperatura Média do Trimestre Mais Frio",
                    12: "Precipitação Anual",
                    13: "Precipitação do Mês Mais Chuvoso",
                    14: "Precipitação do Mês Mais Seco",
                    15: "Sazonalidade da Precipitação (Coeficiente de Variação)",
                    16: "Precipitação do Trimestre Mais Chuvoso",
                    17: "Precipitação do Trimestre Mais Seco",
                    18: "Precipitação do Trimestre Mais Quente",
                    19: "Precipitação do Trimestre Mais Frio"
                },
                'vif': {
                    'limiar': 10
                },
                'earth_engine': {
                    'asset_base': 'projects/earthengine-legacy/assets/projects/sat-io/open-datasets/worldclim/bio'
                }
            }
            
            # Salva a configuração padrão
            with open('config.yaml', 'w') as f:
                yaml.dump(config, f, default_flow_style=False)
            
            logger.info("Arquivo de configuração criado com valores padrão.")
            
        # Carrega a configuração do arquivo
        with open('config.yaml', 'r') as f:
            config = yaml.safe_load(f)
            
        logger.info("Configuração carregada com sucesso.")
        return config
    
    except Exception as e:
        logger.error(f"Erro ao carregar configuração: {str(e)}")
        st.error(f"Erro ao carregar configuração: {str(e)}")
        # Retorna configuração padrão mínima em caso de erro
        return {'vis_params': vis_params_dict, 'descricoes_bio': bio_descriptions_pt}

# Carrega as configurações
CONFIG = carregar_configuracao()

# Extrai dicionários de configuração para uso direto
vis_params_dict = CONFIG['vis_params']
bio_descriptions_pt = CONFIG['descricoes_bio']


# =============================================================================
# Função decoradora para tratamento de exceções
# =============================================================================

def tratar_excecao(func):
    """
    Decorador que captura exceções em funções e registra no log.
    
    Args:
        func: A função a ser decorada.
        
    Returns:
        A função decorada que captura e trata exceções.
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            logger.error(f"Erro em {func.__name__}: {str(e)}", exc_info=True)
            st.error(f"Ocorreu um erro: {str(e)}")
            return None
    return wrapper


# =============================================================================
# Autenticação e Inicialização do Earth Engine
# =============================================================================

@tratar_excecao
@tratar_excecao
def inicializar_ee(forcar_conta_servico=False, mostrar_mensagem=False):
    """
    Inicializa o Google Earth Engine com autenticação.
    
    Args:
        forcar_conta_servico (bool): Se deve forçar o uso da conta de serviço 
                                     mesmo em ambiente de desenvolvimento.
        mostrar_mensagem (bool): Se deve mostrar mensagem de sucesso no Streamlit.
    
    Raises:
        ValueError: Se a informação da conta de serviço estiver incompleta.
        Exception: Qualquer erro durante a inicialização.
    """
    try:
        if forcar_conta_servico or "json_data" in st.secrets:
            with st.spinner("Autenticando Earth Engine...") if mostrar_mensagem else nullcontext():
                json_credentials = st.secrets["json_data"]
                # Se os dados já estiverem como um objeto JSON, você pode usar diretamente
                # Caso contrário, se estiverem como string, use json.loads(json_credentials)
                if isinstance(json_credentials, str):
                    credentials_dict = json.loads(json_credentials)
                else:
                    credentials_dict = json_credentials
                
                if 'client_email' not in credentials_dict:
                    raise ValueError("Informações da conta de serviço não contêm o campo 'client_email'.")
                
                credentials = service_account.Credentials.from_service_account_info(
                    credentials_dict, scopes=oauth.SCOPES
                )
                ee.Initialize(credentials)
                logger.info("Earth Engine inicializado com conta de serviço.")
        else:
            ee.Initialize()
            logger.info("Earth Engine inicializado com autenticação padrão.")
        
        if mostrar_mensagem:
            st.success("Earth Engine inicializado com sucesso!")
    except Exception as e:
        logger.error(f"Falha ao inicializar Earth Engine: {str(e)}", exc_info=True)
        if mostrar_mensagem:
            st.error(f"Não foi possível conectar ao Earth Engine: {str(e)}")
        raise

# Inicializa o Earth Engine (sem mostrar mensagem)
try:
    inicializar_ee(forcar_conta_servico=True, mostrar_mensagem=False)
except Exception as e:
    st.error(f"TAIPA requer conexão com Earth Engine para funcionar corretamente. Erro: {str(e)}")

# =============================================================================
# Funções Auxiliares (Utilitários)
# =============================================================================

@st.cache_data(show_spinner=False)
@tratar_excecao
def carregar_poligono_brasil():
    """
    Carrega o polígono do Brasil usando a biblioteca geobr.
    
    Returns:
        shapely.geometry.Polygon: Polígono representando o território brasileiro.
    """
    with st.spinner("Carregando o mapa do Brasil..."):
        try:
            logger.info("Carregando polígono do Brasil...")
            brazil_gdf = geobr.read_country(year=2019)
            if brazil_gdf.empty:
                raise ValueError("O geobr retornou um DataFrame vazio.")
            
            polygon = brazil_gdf.geometry.unary_union
            logger.info("Polígono do Brasil carregado com sucesso.")
            return polygon
        except Exception as e:
            logger.error(f"Erro ao carregar polígono do Brasil: {str(e)}", exc_info=True)
            st.error(f"Não foi possível carregar o mapa do Brasil: {str(e)}")
            # Retorna um polígono nulo em caso de erro
            return None

poligono_brasil = carregar_poligono_brasil()

@tratar_excecao
def carregar_var_bioclim(numero_var):
    """
    Carrega uma variável bioclimática do Earth Engine.
    
    Args:
        numero_var (int): Número da variável bioclimática (1-19).
        
    Returns:
        ee.Image: Imagem Earth Engine com a variável bioclimática.
    """
    try:
        asset_base = st.secrets["earthengine"]["asset_base"]
        asset_path = f"{asset_base}/wc2_1_5m_bio_{numero_var}"
        logger.info(f"Tentando carregar {asset_path} do projeto")
        return ee.Image(asset_path)
    except Exception as e:
        logger.error(f"Erro ao carregar variável bioclimática {numero_var}: {str(e)}")
        raise ValueError(f"Não foi possível carregar a variável BIO{numero_var}")

@st.cache_data(show_spinner=False)
@tratar_excecao
def extrair_valores_em_cache(_imagem, df, nome_banda, numero_var):
    """
    Extrai valores de uma imagem Earth Engine para pontos em um DataFrame.
    
    Args:
        _imagem (ee.Image): Imagem Earth Engine.
        df (pd.DataFrame): DataFrame com colunas 'latitude' e 'longitude'.
        nome_banda (str): Nome da banda na imagem.
        numero_var (int): Número da variável para fins de log.
        
    Returns:
        list: Lista com os valores extraídos.
    """
    try:
        # Cria uma lista de features a partir dos pontos no DataFrame
        features = [ee.Feature(ee.Geometry.Point([row['longitude'], row['latitude']])) 
                    for _, row in df.iterrows()]
        
        # Converte para FeatureCollection
        fc = ee.FeatureCollection(features)
        
        # Extrai os valores da imagem para cada ponto
        fc_extracted = _imagem.reduceRegions(
            collection=fc,
            reducer=ee.Reducer.first().setOutputs([nome_banda]),
            scale=30
        )
        
        # Obtém os resultados da extração
        dict_extracted = fc_extracted.getInfo()
        
        # Filtra valores None
        values = [f['properties'].get(nome_banda) 
                for f in dict_extracted['features'] if f['properties'].get(nome_banda) is not None]
        
        logger.info(f"Extraídos {len(values)} valores válidos para BIO{numero_var}")
        return values
    except Exception as e:
        logger.error(f"Erro ao extrair valores para BIO{numero_var}: {str(e)}")
        st.warning(f"Erro ao extrair valores para BIO{numero_var}: {str(e)}")
        return []

@tratar_excecao
def converter_shapely_para_ee(shapely_geom):
    """
    Converte uma geometria Shapely para formato Earth Engine.
    
    Args:
        shapely_geom (shapely.geometry.BaseGeometry): Geometria Shapely.
        
    Returns:
        ee.Geometry: Geometria no formato Earth Engine.
    """
    if shapely_geom is None:
        raise ValueError("Geometria Shapely não pode ser None")
        
    geojson = shapely_geom.__geo_interface__
    
    if geojson['type'] == 'Polygon':
        return ee.Geometry.Polygon(geojson['coordinates'])
    elif geojson['type'] == 'MultiPolygon':
        return ee.Geometry.MultiPolygon(geojson['coordinates'])
    else:
        raise ValueError(f"Tipo de geometria não suportado: {geojson['type']}")

def adicionar_camada_ee(self, ee_image_object, vis_params, name):
    """
    Adiciona uma camada Earth Engine a um mapa Folium.
    
    Args:
        self: Instância do mapa Folium.
        ee_image_object (ee.Image): Imagem Earth Engine.
        vis_params (dict): Parâmetros de visualização.
        name (str): Nome da camada.
    """
    try:
        map_id_dict = ee.Image(ee_image_object).getMapId(vis_params)
        folium.raster_layers.TileLayer(
            tiles=map_id_dict['tile_fetcher'].url_format,
            attr='Map Data &copy; Google Earth Engine',
            name=name,
            overlay=True,
            control=True
        ).add_to(self)
    except Exception as e:
        logger.error(f"Erro ao adicionar camada EE ao mapa: {str(e)}")
        # Não levantamos exceção aqui para não interromper o carregamento do mapa

# Adiciona o método ao mapa Folium
folium.Map.add_ee_layer = adicionar_camada_ee


# =============================================================================
# Funções de Geração de Pseudoausências e Busca de Ocorrências
# =============================================================================

@tratar_excecao
def gerar_pseudoausencias_otimizado(presence_df, n_points=100, buffer_distance=0.5):
    """
    Gera pontos de pseudoausência utilizando algoritmo espacialmente otimizado.
    
    Utiliza índice espacial rtree para verificação rápida de contenção e distribui
    pontos de forma mais eficiente dentro da área permitida.
    
    Args:
        presence_df (pd.DataFrame): DataFrame com pontos de presença (precisa ter colunas latitude/longitude).
        n_points (int): Número de pontos de pseudoausência a gerar.
        buffer_distance (float): Distância do buffer em graus.
        
    Returns:
        pd.DataFrame: DataFrame com pontos de pseudoausência.
    """
    try:
        logger.info(f"Gerando {n_points} pseudoausências com buffer de {buffer_distance} graus")
        
        # Converte pontos de presença para geometrias
        presence_points = [Point(lon, lat) for lon, lat in zip(presence_df["longitude"], presence_df["latitude"])]
        
        # Mostra progresso da geração de buffers
        with st.spinner("Gerando buffers ao redor dos pontos de presença..."):
            # Progressbar para visualização do processo
            progress_bar = st.progress(0)
            buffers = []
            
            # Divide em chunks para atualizar a barra de progresso
            chunk_size = max(1, len(presence_points) // 10)
            for i in range(0, len(presence_points), chunk_size):
                chunk = presence_points[i:i+chunk_size]
                buffers.extend([point.buffer(buffer_distance) for point in chunk])
                progress = min(1.0, (i + chunk_size) / len(presence_points))
                progress_bar.progress(progress)
            
            progress_bar.progress(1.0)
        
        # Une todos os buffers e encontra a interseção com o Brasil
        with st.spinner("Calculando área de amostragem (buffers ∩ Brasil)..."):
            union_buffers = unary_union(buffers)
            
            if poligono_brasil is None:
                st.error("Polígono do Brasil não disponível. Usando apenas a união dos buffers.")
                allowed_region = union_buffers
            else:
                allowed_region = union_buffers.intersection(poligono_brasil)
        
        if allowed_region.is_empty:
            st.error("A área permitida (buffers ∩ Brasil) está vazia. Tente diminuir o buffer.")
            return pd.DataFrame()
        
        # Exibe informações sobre a área permitida
        st.write("Limites da área permitida (buffers ∩ Brasil):", allowed_region.bounds)
        st.write("Área permitida (valor em graus²):", allowed_region.area)
        
        # Obtém os limites da área permitida
        minx, miny, maxx, maxy = allowed_region.bounds
        
        # Cria um índice espacial para verificação rápida de contenção
        spatial_index = rtree.index.Index()
        
        # Otimização: pré-calcula células dentro da área permitida
        with st.spinner("Otimizando área de amostragem..."):
            # Divide a área em uma grade
            cell_size = min((maxx - minx), (maxy - miny)) / 20  # Tamanho adaptativo da célula
            valid_cells = []
            
            progress_bar = st.progress(0)
            rows = int((maxy - miny) / cell_size) + 1
            cols = int((maxx - minx) / cell_size) + 1
            
            # Verifica quais células estão dentro da área permitida
            for i in range(rows):
                y = miny + i * cell_size
                for j in range(cols):
                    x = minx + j * cell_size
                    cell_center = Point(x, y)
                    if allowed_region.contains(cell_center):
                        cell_id = len(valid_cells)
                        valid_cells.append((x, y, cell_size))
                        # Adiciona a célula ao índice espacial
                        spatial_index.insert(cell_id, (x, y, x + cell_size, y + cell_size))
                
                progress = min(1.0, (i + 1) / rows)
                progress_bar.progress(progress)
            
            progress_bar.progress(1.0)
        
        if not valid_cells:
            st.error("Não foi possível encontrar células válidas na área permitida.")
            return pd.DataFrame()
        
        # Gera pseudoausências dentro das células válidas
        pseudo_points = []
        attempts = 0
        max_attempts = CONFIG['pseudoausencias']['max_tentativas_por_ponto'] * n_points
        
        with st.spinner(f"Gerando {n_points} pontos de pseudoausência..."):
            progress_bar = st.progress(0)
            
            while len(pseudo_points) < n_points and attempts < max_attempts:
                # Seleciona uma célula aleatória das células válidas
                if valid_cells:
                    cell_idx = np.random.randint(0, len(valid_cells))
                    x_base, y_base, cell_size = valid_cells[cell_idx]
                    
                    # Gera um ponto aleatório dentro da célula
                    random_lon = x_base + np.random.uniform(0, cell_size)
                    random_lat = y_base + np.random.uniform(0, cell_size)
                else:
                    # Fallback: gera dentro dos limites da área permitida
                    random_lon = np.random.uniform(minx, maxx)
                    random_lat = np.random.uniform(miny, maxy)
                
                p = Point(random_lon, random_lat)
                
                # Verifica se o ponto está dentro da área permitida
                if allowed_region.contains(p):
                    pseudo_points.append({
                        "species": "pseudo-absence",
                        "latitude": random_lat,
                        "longitude": random_lon
                    })
                    
                    # Atualiza a barra de progresso
                    progress = min(1.0, len(pseudo_points) / n_points)
                    progress_bar.progress(progress)
                
                attempts += 1
                
                # Feedback a cada 1000 tentativas
                if attempts % 1000 == 0:
                    logger.info(f"Tentativa {attempts}/{max_attempts}, gerados {len(pseudo_points)}/{n_points}")
            
            progress_bar.progress(1.0)
        
        if len(pseudo_points) < n_points:
            st.warning(f"Atingido limite de tentativas. Gerados apenas {len(pseudo_points)}/{n_points} pontos.")
        else:
            st.success(f"Gerados {len(pseudo_points)} pontos de pseudoausência com sucesso.")
        
        logger.info(f"Gerados {len(pseudo_points)} pontos de pseudoausência em {attempts} tentativas")
        return pd.DataFrame(pseudo_points)
    
    except Exception as e:
        logger.error(f"Erro ao gerar pseudoausências: {str(e)}", exc_info=True)
        st.error(f"Erro ao gerar pseudoausências: {str(e)}")
        return pd.DataFrame()

@tratar_excecao
def buscar_ocorrencias_gbif(especie, limite=100):
    """
    Busca ocorrências de uma espécie na API do GBIF.
    
    Implementa tratamento de erros adequado e feedback mais detalhado ao usuário.
    
    Args:
        especie (str): Nome científico da espécie.
        limite (int): Número máximo de ocorrências a buscar.
        
    Returns:
        pd.DataFrame: DataFrame com as ocorrências encontradas.
    """
    try:
        # Validação básica da entrada
        if not especie or not isinstance(especie, str):
            raise ValueError("Nome da espécie deve ser uma string não vazia")
        
        especie = especie.strip()
        if not especie:
            raise ValueError("Nome da espécie não pode ser vazio")
        
        # Verifica limites configurados
        limite_maximo = CONFIG['api']['gbif']['limite_maximo']
        if limite > limite_maximo:
            st.warning(f"Limite ajustado para o máximo permitido: {limite_maximo}")
            limite = limite_maximo
        
        # Prepara a requisição
        url = CONFIG['api']['gbif']['base_url']
        params = {
            "scientificName": especie,
            "hasCoordinate": "true",
            "country": "BR",
            "limit": limite
        }
        
        logger.info(f"Buscando ocorrências para '{especie}' com limite {limite}")
        
        # Faz a requisição com timeout
        with st.spinner(f"Consultando a API do GBIF para {especie}..."):
            response = requests.get(url, params=params, timeout=30)
        
        # Verifica o status da resposta
        if response.status_code == 200:
            data = response.json()
            
            # Verifica se há resultados
            total_results = data.get("count", 0)
            if total_results == 0:
                st.warning(f"Nenhuma ocorrência encontrada para '{especie}' no Brasil.")
                return pd.DataFrame()
            
            # Processa os resultados
            occurrences = []
            for record in data.get("results", []):
                lat = record.get("decimalLatitude")
                lon = record.get("decimalLongitude")
                
                # Valida coordenadas
                if lat is not None and lon is not None:
                    # Verifica se está dentro do Brasil (opcional, já filtramos por país)
                    if poligono_brasil is None or Point(lon, lat).within(poligono_brasil):
                        occurrences.append({
                            "species": especie,
                            "latitude": lat,
                            "longitude": lon,
                            "id_gbif": record.get("key", ""),
                            "data_coleta": record.get("eventDate", "")
                        })
            
            # Cria o DataFrame
            df = pd.DataFrame(occurrences)
            
            # Feedback detalhado ao usuário
            if df.empty:
                st.warning(f"Encontradas {total_results} ocorrências, mas nenhuma com coordenadas válidas.")
            else:
                st.success(f"Encontradas {len(df)} ocorrências com coordenadas válidas (de {total_results} totais).")
            
            logger.info(f"Busca para '{especie}' retornou {len(df)} registros válidos de {total_results} totais")
            return df
        
        elif response.status_code == 429:
            # Limite de requisições excedido
            st.error("Limite de requisições à API do GBIF excedido. Tente novamente mais tarde.")
            logger.warning(f"Rate limit excedido na API do GBIF: {response.text}")
            return pd.DataFrame()
            
        else:
            # Outros erros
            error_msg = f"Erro {response.status_code} ao buscar dados na API do GBIF"
            try:
                error_details = response.json()
                error_msg += f": {error_details.get('message', '')}"
            except:
                error_msg += f": {response.text[:100]}"
            
            st.error(error_msg)
            logger.error(f"Erro na API do GBIF: {error_msg}")
            return pd.DataFrame()
    
    except requests.exceptions.Timeout:
        st.error("Tempo limite excedido ao conectar com a API do GBIF. Tente novamente mais tarde.")
        logger.error("Timeout na requisição à API do GBIF")
        return pd.DataFrame()
        
    except requests.exceptions.ConnectionError:
        st.error("Erro de conexão com a API do GBIF. Verifique sua conexão com a internet.")
        logger.error("Erro de conexão na requisição à API do GBIF")
        return pd.DataFrame()
        
    except Exception as e:
        st.error(f"Erro inesperado ao buscar dados: {str(e)}")
        logger.error(f"Erro ao buscar ocorrências: {str(e)}", exc_info=True)
        return pd.DataFrame()


# =============================================================================
# Funções de Visualização e Extração de Dados
# =============================================================================

@tratar_excecao
def criar_mapa(df, heatmap=False):
    """
    Cria um mapa Folium com pontos de ocorrência e opcionalmente um heatmap.
    
    Args:
        df (pd.DataFrame): DataFrame com pontos (precisa ter colunas latitude/longitude).
        heatmap (bool): Se deve incluir uma camada de heatmap.
        
    Returns:
        folium.Map: Objeto de mapa Folium.
    """
    try:
        # Verifica se o DataFrame tem dados
        if df.empty:
            st.error("Não há dados para exibir no mapa.")
            # Retorna um mapa do Brasil vazio
            m = folium.Map(location=[-15.77, -47.92], zoom_start=4)
            return m
        
        # Calcula os limites e o centro do mapa com base nos dados
        min_lat = df["latitude"].min()
        max_lat = df["latitude"].max()
        min_lon = df["longitude"].min()
        max_lon = df["longitude"].max()
        
        # Adiciona uma margem aos limites (10%)
        lat_margin = (max_lat - min_lat) * 0.1
        lon_margin = (max_lon - min_lon) * 0.1
        min_lat -= lat_margin
        max_lat += lat_margin
        min_lon -= lon_margin
        max_lon += lon_margin
        
        # Define o centro do mapa
        center = [(min_lat + max_lat) / 2, (min_lon + max_lon) / 2]
    
        # Cria o mapa base
        m = folium.Map(location=center)
        
        # Ajusta os limites
        m.fit_bounds([[min_lat, min_lon], [max_lat, max_lon]])
    
        # Adiciona os pontos em um FeatureGroup para que possam ser gerenciados pelo LayerControl
        points_fg = folium.FeatureGroup(name="Ocorrências", show=True)
        
        # Adiciona cada ponto ao mapa
        for _, row in df.iterrows():
            popup_content = f"<b>Espécie:</b> {row['species']}<br><b>Lat:</b> {row['latitude']:.4f}<br><b>Lon:</b> {row['longitude']:.4f}"
            
            # Adiciona informações adicionais se disponíveis
            if 'id_gbif' in row and row['id_gbif']:
                popup_content += f"<br><b>ID GBIF:</b> {row['id_gbif']}"
            if 'data_coleta' in row and row['data_coleta']:
                popup_content += f"<br><b>Data:</b> {row['data_coleta']}"
                
            folium.CircleMarker(
                location=[row["latitude"], row["longitude"]],
                radius=3,
                color="black",
                fill=True,
                fill_color="black",
                fill_opacity=1,
                popup=folium.Popup(popup_content, max_width=300)
            ).add_to(points_fg)
        
        points_fg.add_to(m)
    
        # Adiciona o plugin Draw para permitir que o usuário desenhe polígonos
        draw = Draw(
            export=False,
            draw_options={
                "polygon": True, "polyline": False, "rectangle": False,
                "circle": False, "marker": False, "circlemarker": False
            },
            edit_options={"edit": True}
        )
        draw.add_to(m)
    
        # Se a flag do heatmap estiver ativa, cria um FeatureGroup para o heatmap
        if heatmap:
            heatmap_fg = folium.FeatureGroup(name="Heatmap", show=True)
            pontos = df[["latitude", "longitude"]].values.tolist()
            # Parâmetros ajustados para manchas maiores
            HeatMap(pontos, radius=40, blur=15).add_to(heatmap_fg)
            heatmap_fg.add_to(m)
    
        # Adiciona o controle de layers para permitir a seleção das camadas
        folium.LayerControl().add_to(m)
        return m
        
    except Exception as e:
        logger.error(f"Erro ao criar mapa: {str(e)}", exc_info=True)
        st.error(f"Não foi possível criar o mapa: {str(e)}")
        # Retorna um mapa básico em caso de erro
        return folium.Map(location=[-15.77, -47.92], zoom_start=4)

@st.cache_data(show_spinner=True)
@tratar_excecao
def obter_estatisticas_bioclim(df_occ, df_pseudo, _brasil_ee):
    """
    Extrai os valores das 19 variáveis bioclimáticas e calcula estatísticas descritivas.
    
    Implementa paralelismo para melhorar a performance e feedback detalhado.
    
    Args:
        df_occ (pd.DataFrame): DataFrame com pontos de presença.
        df_pseudo (pd.DataFrame): DataFrame com pontos de pseudoausência.
        _brasil_ee (ee.Geometry): Geometria do Brasil no formato Earth Engine.
        
    Returns:
        tuple: (stats_data, raw_data) contendo estatísticas e valores brutos.
    """
    try:
        stats_data = {}
        raw_data = {}
        stat_names = ['count', 'mean', 'std', 'min', '25%', '50%', '75%', 'max']
        
        # Função auxiliar para extrair uma variável bioclimática
        def extrair_bio(i):
            try:
                bio_name = f"BIO{i}"
                bio_image = carregar_var_bioclim(i)
                
                # Obtém o nome da banda
                band_name = bio_image.bandNames().getInfo()[0]
                
                # Recorta a imagem para o polígono do Brasil
                bio_image_clipped = bio_image.clip(_brasil_ee)
                
                # Extrai os valores para os pontos de presença e pseudoausência
                presence_values = extrair_valores_em_cache(bio_image_clipped, df_occ, band_name, i)
                pseudo_values = extrair_valores_em_cache(bio_image_clipped, df_pseudo, band_name, i)
                combined_values = presence_values + pseudo_values
                
                # Calcula as estatísticas descritivas, ou preenche com NaN se não houver valores
                if combined_values:
                    combined_stats = pd.Series(combined_values).describe()
                else:
                    combined_stats = pd.Series({stat: np.nan for stat in stat_names})
                
                return i, combined_stats, combined_values
            except Exception as e:
                logger.error(f"Erro ao processar BIO{i}: {str(e)}")
                return i, pd.Series({stat: np.nan for stat in stat_names}), []
        
        # Usa executor para paralelismo
        with st.spinner("Extraindo valores das variáveis bioclimáticas..."):
            progresso = st.progress(0)
            
            # Configura o número de workers (ajuste conforme necessário)
            num_workers = min(4, os.cpu_count() or 4)
            
            with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
                futures = {executor.submit(extrair_bio, i): i for i in range(1, 20)}
                
                completados = 0
                total = len(futures)
                
                for future in concurrent.futures.as_completed(futures):
                    i, stats, values = future.result()
                    stats_data[f"BIO{i}"] = stats
                    raw_data[f"BIO{i}"] = values
                    
                    completados += 1
                    progresso.progress(completados / total)
        
        st.success(f"Extração de {total} variáveis bioclimáticas concluída!")
        return stats_data, raw_data
    
    except Exception as e:
        logger.error(f"Erro ao obter estatísticas bioclimáticas: {str(e)}", exc_info=True)
        st.error(f"Erro ao processar variáveis bioclimáticas: {str(e)}")
        return {}, {}

@tratar_excecao
def plotar_densidade(values, label, color):
    """
    Plota a densidade de kernel para um conjunto de valores.
    
    Args:
        values (list): Lista de valores numéricos.
        label (str): Rótulo para a legenda.
        color (str): Cor da linha.
        
    Returns:
        tuple: (xs, density) contendo os valores x e y da curva.
    """
    if not values or len(values) < 5:  # Precisa de pelo menos alguns pontos para KDE
        logger.warning(f"Dados insuficientes para plotar densidade: {len(values) if values else 0} pontos")
        return None, None
    
    try:
        kde = gaussian_kde(values)
        xs = np.linspace(min(values), max(values), 200)
        density = kde(xs)
        plt.plot(xs, density, label=label, color=color)
        return xs, density
    except Exception as e:
        logger.error(f"Erro ao plotar densidade: {str(e)}")
        return None, None

@tratar_excecao
def amostrar_valor_bio(image, lat, lon, band_name, scale=30):
    """
    Amostra um valor de uma imagem Earth Engine em um ponto específico.
    
    Args:
        image (ee.Image): Imagem Earth Engine.
        lat (float): Latitude do ponto.
        lon (float): Longitude do ponto.
        band_name (str): Nome da banda na imagem.
        scale (int): Escala de amostragem em metros.
        
    Returns:
        float: Valor amostrado na posição especificada.
    """
    try:
        pt = ee.Geometry.Point([lon, lat])
        sample = image.sample(pt, scale).first()
        return sample.get(band_name).getInfo()
    except Exception as e:
        logger.error(f"Erro ao amostrar valor: {str(e)}")
        return None


# =============================================================================
# Páginas da Aplicação (Interface com o Usuário)
# =============================================================================

def pagina_inicial():
    """
    Página inicial da aplicação com descrição e instruções.
    """
    st.title("TAIPA - Tecnologia de Aprendizado Interativo em Predição Ambiental")
    
    st.write("""
    **Bem-vindo à TAIPA!**

    A TAIPA é uma plataforma interativa para o ensino de Modelagem de Distribuição de Espécies (SDM)
    utilizando o algoritmo MaxEnt. Aqui, você pode explorar dados de ocorrência, gerar pseudoausências,
    visualizar variáveis ambientais e executar modelos simulados.
    """)
    
    st.info("Utilize o menu lateral para navegar pelas funcionalidades.")
    
    # Exibe detalhes técnicos em um expander
    with st.expander("Informações Técnicas"):
        st.write("""
        A TAIPA utiliza as seguintes tecnologias:
        - Google Earth Engine para acesso a dados ambientais
        - API do GBIF para busca de ocorrências
        - Streamlit para interface com o usuário
        - Folium para visualização de mapas interativos
        - GeoPandas/Shapely para operações espaciais
        
        O fluxo de trabalho recomendado é:
        1. Buscar ocorrências de uma espécie via API
        2. Gerar pontos de pseudoausência
        3. Analisar variáveis ambientais
        4. Executar o modelo MaxEnt
        5. Visualizar resultados e projeções
        """)
    
    # Links úteis
    st.subheader("Links úteis")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("[Documentação GBIF](https://www.gbif.org/developer/summary)")
        st.markdown("[MaxEnt](https://biodiversityinformatics.amnh.org/open_source/maxent/)")
    with col2:
        st.markdown("[Earth Engine](https://developers.google.com/earth-engine)")
        st.markdown("[WorldClim](https://www.worldclim.org/data/bioclim.html)")

def pagina_busca_api():
    """
    Página para busca de ocorrências via API do GBIF.
    """
    st.title("Busca de Ocorrências via API do GBIF")
    st.warning("Os dados da API do GBIF são gratuitos, mas requerem citação. Confira: https://www.gbif.org/citation-guidelines")
    
    # Expander com informações sobre o GBIF
    with st.expander("Sobre o GBIF"):
        st.write("""
        O **Global Biodiversity Information Facility (GBIF)** é uma infraestrutura internacional de dados abertos
        financiada por governos de todo o mundo, destinada a fornecer acesso a dados sobre todas as formas de vida na Terra.
        
        A API utilizada neste aplicativo consulta a base de dados do GBIF para obter registros de ocorrência de
        espécies no Brasil com coordenadas geográficas válidas.
        """)
    
    # Formulário de busca
    with st.form(key="formulario_busca"):
        especie = st.text_input("Digite o nome científico da espécie:")
        
        col1, col2 = st.columns(2)
        with col1:
            limite = st.number_input(
                "Número máximo de ocorrências:",
                min_value=10,
                max_value=CONFIG['api']['gbif']['limite_maximo'],
                value=CONFIG['api']['gbif']['limite_padrao'],
                step=10
            )
        
        with col2:
            st.write("Filtros:")
            apenas_com_coordenadas = st.checkbox("Apenas com coordenadas", value=True, disabled=True)
            apenas_brasil = st.checkbox("Apenas Brasil", value=True, disabled=True)
        
        submitted = st.form_submit_button("Buscar Ocorrências")
        
        if submitted:
            if not especie.strip():
                st.error("Por favor, digite o nome da espécie.")
            else:
                with st.spinner(f"Buscando ocorrências para {especie}..."):
                    df_api = buscar_ocorrencias_gbif(especie, limite)
                
                if df_api is not None:
                    st.session_state.df_api = df_api
                    st.session_state.especie = especie
                    # Limpa a flag do heatmap ao buscar novos dados
                    st.session_state.heatmap_generated = False

    # Se já existem dados de busca, mostra resultados
    if "df_api" in st.session_state:
        df_api = st.session_state.df_api
        
        if df_api.empty:
            st.warning("Nenhuma ocorrência encontrada com os critérios especificados.")
        else:
            st.write(f"**Total de ocorrências retornadas:** {len(df_api)}")
            
            # Tabs para diferentes visualizações
            tab1, tab2, tab3 = st.tabs(["Dados", "Mapa", "Estatísticas"])
            
            with tab1:
                st.write("Visualização dos dados obtidos:")
                st.dataframe(df_api)
                
                # Opção para download
                csv = df_api.to_csv(index=False).encode('utf-8')
                st.download_button(
                    "Download CSV", 
                    csv, 
                    f"ocorrencias_{st.session_state.especie.replace(' ', '_')}.csv",
                    "text/csv",
                    key='download-csv'
                )
            
            with tab2:
                # Controles para o mapa
                st.subheader("Mapa de Ocorrências")
                
                # Checkbox para ativar o heatmap
                heatmap_active = st.checkbox(
                    "Mostrar Heatmap", 
                    value=st.session_state.get("heatmap_generated", False),
                    help="Exibe um mapa de calor das ocorrências"
                )
                
                # Atualiza o estado do heatmap
                st.session_state.heatmap_generated = heatmap_active
                
                # Constrói o mapa
                m = criar_mapa(df_api, heatmap=heatmap_active)
                
                # Renderiza o mapa e captura os dados interativos
                map_data = st_folium(m, width=700, height=500)
                
                # Verifica se o usuário desenhou algum polígono para remoção de pontos
                if map_data and map_data.get("all_drawings"):
                    polygon_features = [
                        feature for feature in map_data["all_drawings"]
                        if feature.get("geometry", {}).get("type") == "Polygon"
                    ]
                    
                    if polygon_features:
                        st.info(f"{len(polygon_features)} polígono(s) desenhado(s) detectado(s).")
                        
                        if st.button("Remover pontos dentro do(s) polígono(s)"):
                            total_removidos = 0
                            
                            for poly_feature in polygon_features:
                                poly_coords = poly_feature["geometry"]["coordinates"][0]
                                polygon_shapely = Polygon(poly_coords)
                                
                                indices_to_remove = []
                                for idx, row in df_api.iterrows():
                                    point = Point(row["longitude"], row["latitude"])
                                    if polygon_shapely.contains(point):
                                        indices_to_remove.append(idx)
                                
                                total_removidos += len(indices_to_remove)
                                
                                if indices_to_remove:
                                    df_api = df_api.drop(indices_to_remove).reset_index(drop=True)
                            
                            if total_removidos > 0:
                                st.session_state.df_api = df_api
                                st.success(f"{total_removidos} ponto(s) removido(s) dentro do(s) polígono(s).")
                                
                                # Atualiza o mapa
                                st.rerun()
                            else:
                                st.info("Nenhum ponto encontrado dentro do(s) polígono(s).")
            
            with tab3:
                st.subheader("Estatísticas Básicas")
                
                if len(df_api) > 0:
                    # Estatísticas de localização
                    st.write("**Estatísticas de Localização:**")
                    
                    # Divide em colunas
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.write("Latitude:")
                        st.write(f"- Mínima: {df_api['latitude'].min():.4f}°")
                        st.write(f"- Máxima: {df_api['latitude'].max():.4f}°")
                        st.write(f"- Média: {df_api['latitude'].mean():.4f}°")
                    
                    with col2:
                        st.write("Longitude:")
                        st.write(f"- Mínima: {df_api['longitude'].min():.4f}°")
                        st.write(f"- Máxima: {df_api['longitude'].max():.4f}°")
                        st.write(f"- Média: {df_api['longitude'].mean():.4f}°")
                    
                    # Distância máxima entre pontos (calculada de forma aproximada)
                    dist_lat = df_api['latitude'].max() - df_api['latitude'].min()
                    dist_lon = df_api['longitude'].max() - df_api['longitude'].min()
                    dist_km = ((dist_lat * 111) ** 2 + (dist_lon * 111) ** 2) ** 0.5
                    
                    st.write(f"**Distância máxima entre pontos:** {dist_km:.2f} km (aproximada)")
                    
                    if 'data_coleta' in df_api.columns and df_api['data_coleta'].notna().any():
                        # Tenta converter para datetime
                        try:
                            df_api['data_coleta'] = pd.to_datetime(df_api['data_coleta'])
                            anos = df_api['data_coleta'].dt.year
                            
                            st.write("**Distribuição temporal:**")
                            st.write(f"- Ano mais antigo: {anos.min()}")
                            st.write(f"- Ano mais recente: {anos.max()}")
                            
                            # Histograma de anos
                            fig, ax = plt.subplots()
                            anos.hist(bins=min(20, anos.nunique()), ax=ax)
                            ax.set_xlabel('Ano')
                            ax.set_ylabel('Número de ocorrências')
                            ax.set_title('Distribuição temporal das ocorrências')
                            st.pyplot(fig)
                        except:
                            st.write("Não foi possível analisar dados temporais.")
                else:
                    st.write("Sem dados suficientes para gerar estatísticas.")

def pagina_pseudoausencias():
    """
    Página para geração de pseudoausências.
    """
    st.title("Geração de Pseudoausências")
    st.write("Gera pontos de pseudoausência utilizando buffers dos pontos de presença (limitado ao Brasil).")
    
    # Expander com explicação sobre pseudoausências
    with st.expander("O que são pseudoausências?"):
        st.write("""
        **Pseudoausências** são pontos artificiais que representam locais onde a espécie provavelmente 
        não ocorre. Eles são necessários para algoritmos como o MaxEnt que requerem dados de presença e ausência.
        
        Nesta implementação, as pseudoausências são geradas:
        1. Criando um buffer ao redor de cada ponto de presença
        2. Unindo todos os buffers para criar uma área de amostragem
        3. Intersectando esta área com o polígono do Brasil
        4. Gerando pontos aleatórios dentro desta área
        
        Isso garante que as pseudoausências estejam em áreas ambientalmente similares às presenças,
        mas não muito próximas a pontos de ocorrência conhecidos.
        """)
    
    if "df_api" in st.session_state:
        df_presence = st.session_state.df_api
        
        if df_presence.empty:
            st.error("Os dados de presença estão vazios. Faça uma nova busca de ocorrências.")
            return
        
        st.write(f"**Dados de presença disponíveis:** {len(df_presence)} pontos")
        
        with st.form(key="formulario_pseudoausencia"):
            col1, col2 = st.columns(2)
            
            with col1:
                n_points = st.slider(
                    "Número de pseudoausências a gerar", 
                    min_value=CONFIG['pseudoausencias']['minimo'], 
                    max_value=CONFIG['pseudoausencias']['maximo'], 
                    value=CONFIG['pseudoausencias']['padrao'], 
                    step=10
                )
            
            with col2:
                # Slider para buffer em Km
                buffer_distance_km = st.slider(
                    "Tamanho do buffer (em Km)", 
                    min_value=1, 
                    max_value=200, 
                    value=CONFIG['pseudoausencias']['buffer_padrao_km'], 
                    step=1
                )
            
            # Converte de Km para graus (1 grau ≈ 111 Km)
            buffer_distance_degrees = buffer_distance_km / 111.0
            
            submitted = st.form_submit_button("Gerar Pseudoausências")
            
            if submitted:
                pseudo_df = gerar_pseudoausencias_otimizado(df_presence, n_points, buffer_distance_degrees)
                
                if not pseudo_df.empty:
                    st.session_state.df_pseudo = pseudo_df
                    st.success(f"{len(pseudo_df)} pontos de pseudoausência gerados com sucesso.")
        
        if "df_pseudo" in st.session_state:
            pseudo_df = st.session_state.df_pseudo
            
            if not pseudo_df.empty:
                st.subheader("Visualização das Pseudoausências")
                
                # Tabs para diferentes visualizações
                tab1, tab2 = st.tabs(["Dados", "Mapa"])
                
                with tab1:
                    st.write("Primeiros registros das pseudoausências:")
                    st.dataframe(pseudo_df.head())
                    
                    # Opção para download
                    csv = pseudo_df.to_csv(index=False).encode('utf-8')
                    st.download_button(
                        "Download CSV", 
                        csv, 
                        f"pseudoausencias_{st.session_state.especie.replace(' ', '_')}.csv",
                        "text/csv",
                        key='download-pseudo-csv'
                    )
                
                with tab2:
                    # Preparar DataFrames para visualização no mapa
                    df_presence_viz = df_presence.copy()
                    df_presence_viz["marker_color"] = "blue"  # Presenças em azul
                    
                    pseudo_df_viz = pseudo_df.copy()
                    pseudo_df_viz["marker_color"] = "red"  # Pseudoausências em vermelho
                    
                    # Unir os DataFrames para visualização
                    df_combined = pd.concat([df_presence_viz, pseudo_df_viz])
                    
                    # Criar o mapa
                    min_lat = df_combined["latitude"].min()
                    max_lat = df_combined["latitude"].max()
                    min_lon = df_combined["longitude"].min()
                    max_lon = df_combined["longitude"].max()
                    center = [(min_lat + max_lat) / 2, (min_lon + max_lon) / 2]
                    
                    m = folium.Map(location=center)
                    m.fit_bounds([[min_lat, min_lon], [max_lat, max_lon]])
                    
                    # Adicionar presenças (azul)
                    presence_group = folium.FeatureGroup(name="Presenças")
                    for _, row in df_presence.iterrows():
                        folium.CircleMarker(
                            location=[row["latitude"], row["longitude"]],
                            radius=3,
                            color="blue",
                            fill=True,
                            fill_color="blue",
                            fill_opacity=1,
                            popup=f"Presença: {row['species']}"
                        ).add_to(presence_group)
                    presence_group.add_to(m)
                    
                    # Adicionar pseudoausências (vermelho)
                    pseudo_group = folium.FeatureGroup(name="Pseudoausências")
                    for _, row in pseudo_df.iterrows():
                        folium.CircleMarker(
                            location=[row["latitude"], row["longitude"]],
                            radius=3,
                            color="red",
                            fill=True,
                            fill_color="red",
                            fill_opacity=1,
                            popup="Pseudoausência"
                        ).add_to(pseudo_group)
                    pseudo_group.add_to(m)
                    
                    # Adicionar controle de camadas
                    folium.LayerControl().add_to(m)
                    
                    # Exibir o mapa
                    st_folium(m, width=700, height=500)
    else:
        st.warning("Dados de presença não encontrados. Execute a busca via API primeiro.")
        
        # Link para a página de busca
        st.markdown("[Ir para a página de busca](#Busca-Ocorrência)")


def pagina_variaveis_ambientais():
    """
    Página para análise de variáveis ambientais.
    """
    st.title("Estatísticas Descritivas - Bioclima")
    
    # Expander com informações sobre variáveis bioclimáticas
    with st.expander("Sobre Variáveis Bioclimáticas"):
        st.write("""
        As **variáveis bioclimáticas** são derivadas de dados mensais de temperatura e precipitação. 
        Representam tendências anuais, sazonalidade e fatores ambientais extremos que são relevantes 
        para a distribuição de espécies.
        
        Estas 19 variáveis são frequentemente utilizadas em modelagem de nicho ecológico 
        e distribuição de espécies por capturarem aspectos importantes do clima que influenciam 
        a sobrevivência e reprodução dos organismos.
        """)

    # Tabela explicativa das variáveis BIO
    bio_desc_df = pd.DataFrame(bio_descriptions_pt.items(), columns=["BIO", "Significado"])
    st.subheader("Significado das Variáveis BIO")
    st.dataframe(bio_desc_df)
    
    # Verifica se os dados de presença e pseudoausência estão disponíveis
    if "df_api" not in st.session_state:
        st.warning("Dados de presença não encontrados. Execute a busca via API primeiro.")
        return
    if "df_pseudo" not in st.session_state:
        st.warning("Dados de pseudoausência não encontrados. Gere pseudoausências primeiro.")
        return

    df_occ = st.session_state.df_api
    df_pseudo = st.session_state.df_pseudo
    
    # Converte o polígono do Brasil para formato Earth Engine
    try:
        brazil_ee = converter_shapely_para_ee(poligono_brasil)
        if brazil_ee is None:
            st.error("Não foi possível converter o polígono do Brasil.")
            return
    except Exception as e:
        st.error(f"Erro ao preparar o polígono do Brasil: {str(e)}")
        return

    # Usa uma função cacheada para extrair os dados das variáveis bioclimáticas
    with st.spinner("Extraindo valores das variáveis bioclimáticas..."):
        stats_data, raw_data = obter_estatisticas_bioclim(df_occ, df_pseudo, brazil_ee)
        
        if not stats_data or not raw_data:
            st.error("Não foi possível obter dados das variáveis bioclimáticas.")
            return

    # Exibe as estatísticas descritivas dos dados combinados
    df_stats = pd.DataFrame(stats_data).T
    
    st.subheader("Estatísticas Descritivas")
    st.dataframe(df_stats)
    
    # Opção para download das estatísticas
    csv_stats = df_stats.to_csv().encode('utf-8')
    st.download_button(
        "Download Estatísticas CSV", 
        csv_stats, 
        "estatisticas_bioclim.csv",
        "text/csv",
        key='download-stats-csv'
    )

    # Cria DataFrame com os valores extraídos para correlação e VIF
    raw_data_df = pd.DataFrame({k: pd.Series(v) for k, v in raw_data.items()})
    
    # Análise de correlação
    with st.spinner("Calculando matriz de correlação..."):
        corr_matrix = raw_data_df.corr()
        
        st.subheader("Matriz de Correlação Pairwise")
        #st.dataframe(corr_matrix.style.background_gradient(cmap="coolwarm"))
        
        # Exibe o heatmap da matriz de correlação
        fig, ax = plt.subplots(figsize=(12, 10))
        sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", square=True, fmt=".2f", ax=ax, annot_kws={"size": 8})
        ax.set_title("Matriz de Correlação entre Variáveis Bioclimáticas")
        st.pyplot(fig)
    
    # Cálculo do VIF incluindo o intercepto (constante)
    with st.spinner("Calculando VIF para todas as variáveis..."):
        try:
            raw_data_df_clean = raw_data_df.dropna()
            if raw_data_df_clean.empty:
                st.warning("Dados insuficientes para cálculo do VIF após remoção de NaNs.")
                return
            
            # Adiciona a constante
            X = add_constant(raw_data_df_clean)
            vif_all = pd.DataFrame()
            vif_all["Feature"] = X.columns
            vif_all["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
            
            st.subheader("VIF - Valores Iniciais para Todas as Variáveis")
            st.dataframe(vif_all)
        except Exception as e:
            st.error(f"Erro ao calcular VIF inicial: {str(e)}")
            logger.error(f"Erro ao calcular VIF: {str(e)}", exc_info=True)
    
    # Eliminação stepwise: remoção das variáveis com maior VIF (excluindo a constante)
    with st.spinner("Realizando eliminação stepwise de variáveis com alto VIF..."):
        try:
            threshold = CONFIG['vif']['limiar']
            variables = list(raw_data_df_clean.columns)
            
            # Expander para mostrar o processo detalhado
            with st.expander("Ver processo de eliminação stepwise"):
                iteration = 1
                
                while True:
                    # Adiciona constante para o cálculo
                    X_temp = add_constant(raw_data_df_clean[variables])
                    vif_df = pd.DataFrame()
                    vif_df["Feature"] = X_temp.columns
                    vif_df["VIF"] = [variance_inflation_factor(X_temp.values, i) for i in range(X_temp.shape[1])]
                    
                    # Exclui a constante da análise para remoção
                    vif_df_no_const = vif_df[vif_df["Feature"] != "const"]
                    max_vif = vif_df_no_const["VIF"].max()
                    
                    # Exibe a tabela da iteração atual
                    st.write(f"**Iteração {iteration}**")
                    st.dataframe(vif_df_no_const)
                    
                    if max_vif < threshold or len(variables) <= 1:
                        st.write(f"**Processo concluído** - Todas as variáveis com VIF < {threshold} ou restou apenas 1 variável")
                        break
                        
                    max_var = vif_df_no_const.sort_values("VIF", ascending=False)["Feature"].iloc[0]
                    max_vif_value = vif_df_no_const.sort_values("VIF", ascending=False)["VIF"].iloc[0]
                    
                    st.write(f"Removendo variável com maior VIF: **{max_var}** (VIF = {max_vif_value:.2f})")
                    variables.remove(max_var)
                    
                    iteration += 1
            
            # Calcula VIF final
            X_final = add_constant(raw_data_df_clean[variables])
            final_vif = pd.DataFrame()
            final_vif["Feature"] = X_final.columns
            final_vif["VIF"] = [variance_inflation_factor(X_final.values, i) for i in range(X_final.shape[1])]
            
            # Remove a constante para exibição
            final_vif = final_vif[final_vif["Feature"] != "const"]
            
            st.subheader("VIF Final Após Eliminação Stepwise")
            st.dataframe(final_vif)
            st.success(f"Variáveis selecionadas para modelagem (VIF < {threshold}): {', '.join(variables)}")
            
        except Exception as e:
            st.error(f"Erro ao realizar eliminação stepwise: {str(e)}")
            logger.error(f"Erro em eliminação stepwise: {str(e)}", exc_info=True)

    # Permite que o usuário selecione as variáveis para visualização
    selected_vars = st.multiselect(
        "Selecione as variáveis para visualizar no mapa:",
        options=variables if 'variables' in locals() else raw_data_df.columns,
        default=variables[:3] if 'variables' in locals() else raw_data_df.columns[:3]
    )

    if selected_vars:
        with st.spinner("Preparando visualização das variáveis selecionadas..."):
            try:
                # Cria um mapa centrado no polígono do Brasil
                bounds = poligono_brasil.bounds
                m = folium.Map(tiles="OpenStreetMap")
                m.fit_bounds([[bounds[1], bounds[0]], [bounds[3], bounds[2]]])
            
                # Para cada variável selecionada, adiciona uma camada do Earth Engine
                for var in selected_vars:
                    # Extrai o número da variável (por exemplo, de "BIO3" extrai 3)
                    var_number = int(var.replace("BIO", ""))
                    
                    # Carrega e processa a variável
                    try:
                        bio_image = carregar_var_bioclim(var_number)
                        band_name = bio_image.bandNames().getInfo()[0]
                        
                        # Recorta a imagem para o polígono do Brasil
                        bio_image_clipped = bio_image.clip(converter_shapely_para_ee(poligono_brasil))
                        
                        # Utiliza os parâmetros de visualização, se disponíveis
                        vis_params = vis_params_dict.get(var_number, {})
                        
                        # Adiciona a camada ao mapa
                        m.add_ee_layer(
                            bio_image_clipped, 
                            vis_params, 
                            f"{var} - {bio_descriptions_pt[var_number]}"
                        )
                    except Exception as e:
                        st.warning(f"Erro ao adicionar {var} ao mapa: {str(e)}")
                        continue
            
                # Adiciona os pontos de presença (preto)
                presence_group = folium.FeatureGroup(name="Presenças")
                for _, row in df_occ.iterrows():
                    folium.CircleMarker(
                        location=[row["latitude"], row["longitude"]],
                        radius=3,
                        color="black",
                        fill=True,
                        fill_color="black",
                        fill_opacity=1,
                        popup=f"Presença: {row['species']}"
                    ).add_to(presence_group)
                presence_group.add_to(m)
            
                # Adiciona os pontos de pseudoausência (vermelho)
                pseudo_group = folium.FeatureGroup(name="Pseudoausências")
                for _, row in df_pseudo.iterrows():
                    folium.CircleMarker(
                        location=[row["latitude"], row["longitude"]],
                        radius=3,
                        color="red",
                        fill=True,
                        fill_color="red",
                        fill_opacity=1,
                        popup="Pseudoausência"
                    ).add_to(pseudo_group)
                pseudo_group.add_to(m)
            
                # Adiciona um controle de camadas para que o usuário possa ativar/desativar as variáveis
                folium.LayerControl().add_to(m)
            
                st.subheader("Visualização das Variáveis Selecionadas no Brasil")
                st_folium(m, width=700, height=500)
            
            except Exception as e:
                st.error(f"Erro ao criar mapa: {str(e)}")
                logger.error(f"Erro ao criar mapa de variáveis: {str(e)}", exc_info=True)

def executar_modelo():
    """
    Página para execução do modelo MaxEnt.
    Obs: Esta implementação atual é apenas simulada.
    """
    st.title("Execução do Modelo MaxEnt")
    st.warning("AVISO: Esta seção está em desenvolvimento. Os resultados são simulados.")
    
    # Verifica disponibilidade de dados
    if "df_api" not in st.session_state:
        st.warning("Dados de presença não encontrados. Execute a busca via API primeiro.")
        return
    if "df_pseudo" not in st.session_state:
        st.warning("Dados de pseudoausência não encontrados. Gere pseudoausências primeiro.")
        return
    
    st.write(f"**Pontos de presença disponíveis:** {len(st.session_state.df_api)}")
    st.write(f"**Pontos de pseudoausência disponíveis:** {len(st.session_state.df_pseudo)}")
    
    # Formulário de configuração do modelo
    with st.form(key="config_modelo"):
        st.subheader("Configuração do Modelo")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Parâmetros básicos
            regularizacao = st.slider("Multiplicador de Regularização", 0.1, 5.0, 1.0, 0.1,
                                    help="Controla o overfitting. Valores maiores = modelos mais simples")
            max_iteracoes = st.slider("Iterações Máximas", 100, 5000, 500, 100,
                                    help="Número máximo de iterações de treinamento")
            
        with col2:
            # Divisão de dados
            proporcao_teste = st.slider("Proporção para Teste", 0.0, 0.5, 0.25, 0.05,
                                        help="Fração dos dados a serem reservados para teste")
            
            # Semente aleatória
            semente = st.number_input("Semente Aleatória", 1, 9999, 42,
                                    help="Semente para reprodutibilidade")
        
        # Features complexas
        features = st.multiselect(
            "Features a incluir",
            options=["Linear", "Quadrática", "Produto", "Threshold", "Hinge"],
            default=["Linear", "Quadrática", "Hinge"],
            help="Tipos de transformações de variáveis a incluir no modelo"
        )
        
        submitted = st.form_submit_button("Executar Modelo")
        
        if submitted:
            with st.spinner("Processando o modelo..."):
                # Simulação de processamento
                progress_bar = st.progress(0)
                for i in range(100):
                    time.sleep(0.02)  # Simula processamento
                    progress_bar.progress((i + 1) / 100)
                
                # Cria resultados simulados
                simulated_map = np.random.rand(100, 100)
                
                # Aumenta a probabilidade próximo ao centro para simular um padrão
                rows, cols = simulated_map.shape
                for i in range(rows):
                    for j in range(cols):
                        # Distância normalizada do centro
                        dist = ((i - rows/2)**2 + (j - cols/2)**2)**0.5 / (0.5 * (rows**2 + cols**2)**0.5)
                        # Ajusta a probabilidade: mais alta no centro, mais baixa nas bordas
                        simulated_map[i, j] = simulated_map[i, j] * (1 - 0.7 * dist)
                
                # Gera AUC simulada
                auc_train = 0.75 + 0.2 * np.random.random()
                auc_test = auc_train - 0.05 - 0.1 * np.random.random()
                
                # Armazena resultados
                st.session_state["model_results"] = {
                    "map": simulated_map,
                    "auc_train": auc_train,
                    "auc_test": auc_test,
                    "params": {
                        "regularization": regularizacao,
                        "max_iterations": max_iteracoes,
                        "test_proportion": proporcao_teste,
                        "seed": semente,
                        "features": features
                    }
                }
                
                st.success("Modelo executado com sucesso!")
    
    # Exibe os resultados se disponíveis
    if "model_results" in st.session_state:
        results = st.session_state["model_results"]
        
        st.subheader("Resultados do Modelo")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            # Mapa de probabilidade
            st.write("### Mapa de Probabilidade")
            fig, ax = plt.subplots(figsize=(8, 6))
            cax = ax.imshow(results["map"], cmap="viridis", interpolation='nearest')
            ax.set_title("Mapa de Probabilidade de Ocorrência (Simulado)")
            fig.colorbar(cax, label="Probabilidade")
            st.pyplot(fig)
        
        with col2:
            # Métricas
            st.write("### Métricas de Desempenho")
            st.write(f"**AUC (Treino):** {results['auc_train']:.3f}")
            st.write(f"**AUC (Teste):** {results['auc_test']:.3f}")
            
            # Configurações usadas
            st.write("### Configurações")
            for param, value in results["params"].items():
                st.write(f"**{param}:** {value}")
        
        # Curva de resposta simulada
        st.subheader("Curvas de Resposta")
        
        # Gera curvas de resposta simuladas para algumas variáveis
        variables = ["BIO1", "BIO12", "BIO15"]
        
        # Cria um gráfico com subplots para as curvas de resposta
        fig, axes = plt.subplots(1, len(variables), figsize=(15, 4))
        
        for i, var in enumerate(variables):
            x = np.linspace(0, 1, 100)
            
            # Gera diferentes formas de curva para cada variável
            if i == 0:  # Unimodal
                y = np.exp(-((x - 0.6) ** 2) / (2 * 0.15 ** 2))
            elif i == 1:  # Sigmoidal
                y = 1 / (1 + np.exp(-10 * (x - 0.5)))
            else:  # Linear com ruído
                y = 0.3 + 0.6 * x + 0.1 * np.random.randn(100)
                y = np.clip(y, 0, 1)
            
            axes[i].plot(x, y, 'b-')
            axes[i].set_xlabel(f"Valor de {var} (normalizado)")
            axes[i].set_ylabel("Probabilidade")
            axes[i].set_title(f"Resposta para {var}")
            axes[i].grid(True, linestyle='--', alpha=0.7)
        
        plt.tight_layout()
        st.pyplot(fig)
        
        # Salvar o modelo
        with st.expander("Salvar Modelo"):
            nome_modelo = st.text_input("Nome do Modelo", 
                                      value=f"Modelo_{st.session_state.get('especie', 'especie').replace(' ', '_')}")
            if st.button("Salvar"):
                st.success(f"Modelo '{nome_modelo}' salvo com sucesso (simulado).")
                st.info("Em uma implementação completa, esta funcionalidade permitiria baixar o modelo para uso posterior.")

def resultados():
    """
    Página para visualização de resultados.
    """
    st.title("Resultados do Modelo")
    st.warning("AVISO: Esta seção está em desenvolvimento. Os resultados são simulados.")
    
    if "model_results" in st.session_state:
        results = st.session_state["model_results"]
        
        # Tabs para diferentes visualizações
        tabs = st.tabs(["Mapa de Probabilidade", "Estatísticas", "Avaliação", "Download"])
        
        with tabs[0]:
            st.subheader("Mapa de Distribuição Potencial")
            
            # Controles de visualização
            threshold = st.slider(
                "Limiar de Corte (threshold)", 
                0.0, 1.0, 0.5, 0.01,
                help="Valor de corte para presença/ausência"
            )
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Mapa de probabilidade contínua
                fig1, ax1 = plt.subplots(figsize=(8, 6))
                cax = ax1.imshow(results["map"], cmap="viridis", interpolation='nearest')
                ax1.set_title("Probabilidade de Ocorrência")
                fig1.colorbar(cax, label="Probabilidade")
                st.pyplot(fig1)
            
            with col2:
                # Mapa binário (acima/abaixo do threshold)
                binary_map = results["map"] >= threshold
                fig2, ax2 = plt.subplots(figsize=(8, 6))
                cax2 = ax2.imshow(binary_map, cmap="Greens", interpolation='nearest')
                ax2.set_title(f"Presença/Ausência (Limiar: {threshold:.2f})")
                st.pyplot(fig2)
            
            # Estatísticas da previsão
            proporção_prevista = np.mean(binary_map)
            st.write(f"**Proporção da área prevista como adequada:** {proporção_prevista:.2%}")
        
        with tabs[1]:
            st.subheader("Estatísticas do Modelo")
            
            # Distribuição de valores de predição (histograma)
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.histplot(results["map"].flatten(), bins=20, kde=True, ax=ax)
            ax.set_xlabel("Probabilidade Prevista")
            ax.set_ylabel("Frequência")
            ax.set_title("Distribuição de Valores de Predição")
            st.pyplot(fig)
            
            # Estatísticas descritivas
            st.write("**Estatísticas dos Valores Previstos:**")
            pred_values = results["map"].flatten()
            stats = {
                "Mínimo": np.min(pred_values),
                "1º Quartil": np.percentile(pred_values, 25),
                "Mediana": np.median(pred_values),
                "Média": np.mean(pred_values),
                "3º Quartil": np.percentile(pred_values, 75),
                "Máximo": np.max(pred_values),
                "Desvio Padrão": np.std(pred_values)
            }
            
            st.dataframe(pd.DataFrame([stats]).T.rename(columns={0: "Valor"}))
        
        with tabs[2]:
            st.subheader("Avaliação do Modelo")
            
            # ROC Curve (simulada)
            st.write("### Curva ROC")
            
            fpr = np.linspace(0, 1, 100)
            tpr_train = fpr ** (1 / (results["auc_train"] * 5))  # Simulação de curva ROC
            tpr_test = fpr ** (1 / (results["auc_test"] * 5))
            
            fig, ax = plt.subplots(figsize=(8, 6))
            ax.plot(fpr, tpr_train, 'b-', label=f'Treino (AUC = {results["auc_train"]:.3f})')
            ax.plot(fpr, tpr_test, 'r-', label=f'Teste (AUC = {results["auc_test"]:.3f})')
            ax.plot([0, 1], [0, 1], 'k--', label='Aleatório (AUC = 0.5)')
            ax.set_xlabel('Taxa de Falsos Positivos')
            ax.set_ylabel('Taxa de Verdadeiros Positivos')
            ax.set_title('Curva ROC')
            ax.legend()
            ax.grid(True, linestyle='--', alpha=0.7)
            st.pyplot(fig)
            
            # Tabela de importância de variáveis (simulada)
            st.write("### Importância de Variáveis")
            
            # Cria dados simulados de importância
            var_names = ["BIO1", "BIO12", "BIO15", "BIO4", "BIO7"]
            importance = np.random.rand(len(var_names))
            importance = importance / importance.sum() * 100  # Normaliza para percentual
            
            var_import_df = pd.DataFrame({
                'Variável': var_names,
                'Importância (%)': importance,
                'Contribuição': importance,
                'Permutação': importance * (0.8 + 0.4 * np.random.rand(len(var_names)))
            })
            
            var_import_df = var_import_df.sort_values('Importância (%)', ascending=False)
            var_import_df['Importância (%)'] = var_import_df['Importância (%)'].round(2)
            var_import_df['Contribuição'] = var_import_df['Contribuição'].round(2)
            var_import_df['Permutação'] = var_import_df['Permutação'].round(2)
            
            st.dataframe(var_import_df)
            
            # Gráfico de barras de importância
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.barplot(x='Importância (%)', y='Variável', data=var_import_df, ax=ax)
            ax.set_title('Importância das Variáveis')
            ax.set_xlabel('Importância (%)')
            st.pyplot(fig)
        
        with tabs[3]:
            st.subheader("Download de Resultados")
            
            st.write("**Em uma implementação completa, seriam disponibilizadas opções para download de:**")
            
            st.write("- Arquivo do modelo (formato .asc)")
            st.write("- Relatório em PDF com detalhes da modelagem")
            st.write("- Mapas em formato GeoTIFF")
            st.write("- Tabelas de resultados em CSV")
            
            # Botões de download simulados
            col1, col2 = st.columns(2)
            with col1:
                if st.button("Download do Mapa (simulado)"):
                    st.success("Download iniciado (simulação)")
            
            with col2:
                if st.button("Download do Relatório (simulado)"):
                    st.success("Relatório gerado (simulação)")
    else:
        st.warning("Nenhum resultado encontrado. Execute o modelo na seção 'Executar Modelo'.")
        
        # Link para a página de execução
        st.markdown("[Ir para Executar Modelo](#Executar-Modelo)")

def projecao_futura():
    """
    Página para projeção futura.
    """
    st.title("Projeção Futura")
    st.warning("AVISO: Esta seção está em desenvolvimento. As projeções são simuladas.")
    
    if "model_results" not in st.session_state:
        st.warning("Nenhum modelo encontrado. Execute o modelo na seção 'Executar Modelo'.")
        return
    
    # Opções de cenários climáticos
    st.subheader("Configuração da Projeção")
    
    col1, col2 = st.columns(2)
    
    with col1:
        periodo = st.selectbox(
            "Período Futuro",
            options=["2041-2060 (2050)", "2061-2080 (2070)", "2081-2100 (2090)"],
            index=0
        )
    
    with col2:
        cenario = st.selectbox(
            "Cenário de Emissão",
            options=["SSP1-2.6 (Otimista)", "SSP2-4.5 (Intermediário)", "SSP5-8.5 (Pessimista)"],
            index=1
        )
    
    # Modelos climáticos
    modelo_climatico = st.selectbox(
        "Modelo Climático Global",
        options=["Média dos modelos", "CMIP6-ACESS", "CMIP6-MPI-ESM", "CMIP6-NCAR"],
        index=0
    )
    
    # Execução da projeção
    if st.button("Executar Projeção"):
        with st.spinner(f"Processando projeção para {periodo}, cenário {cenario}..."):
            # Simulação de processamento
            progress_bar = st.progress(0)
            for i in range(100):
                time.sleep(0.02)  # Simula processamento
                progress_bar.progress((i + 1) / 100)
            
            # Cria um mapa futuro simulado baseado no mapa atual, mas com alterações
            base_map = st.session_state["model_results"]["map"]
            
            # Fatores de alteração baseados no cenário selecionado
            if "SSP1-2.6" in cenario:
                # Cenário otimista: pequena redução na área adequada
                factor = 0.9
                shift = 0.1  # Deslocamento para o sul
            elif "SSP2-4.5" in cenario:
                # Cenário intermediário: redução moderada
                factor = 0.7
                shift = 0.2
            else:
                # Cenário pessimista: grande redução
                factor = 0.5
                shift = 0.3
            
            # Adiciona um fator temporal
            if "2050" in periodo:
                time_factor = 1.0
            elif "2070" in periodo:
                time_factor = 1.5
            else:  # 2090
                time_factor = 2.0
            
            # Cria o mapa futuro com deslocamento e alteração na intensidade
            rows, cols = base_map.shape
            simulated_future_map = np.zeros_like(base_map)
            
            # Aplica transformação no mapa
            shift_pixels = int(shift * rows * time_factor)
            for i in range(rows):
                for j in range(cols):
                    # Índice de origem com deslocamento para o sul/pólos
                    orig_i = max(0, min(rows-1, i - shift_pixels))
                    # Reduz a intensidade com base nos fatores
                    reduction = factor / time_factor
                    simulated_future_map[i, j] = base_map[orig_i, j] * reduction
            
            # Adiciona alguma variação aleatória
            noise = np.random.normal(0, 0.05, size=simulated_future_map.shape)
            simulated_future_map = np.clip(simulated_future_map + noise, 0, 1)
            
            # Armazena os resultados da projeção
            st.session_state["future_projection"] = {
                "map": simulated_future_map,
                "base_map": base_map,
                "params": {
                    "period": periodo,
                    "scenario": cenario,
                    "model": modelo_climatico
                }
            }
            
            st.success("Projeção realizada com sucesso!")
        
        # Exibe os resultados da projeção
        if "future_projection" in st.session_state:
            proj = st.session_state["future_projection"]
            
            st.subheader("Resultados da Projeção")
            
            # Tabs para diferentes visualizações
            proj_tabs = st.tabs(["Comparação", "Mudanças", "Estatísticas"])
            
            with proj_tabs[0]:
                st.write("### Comparação: Presente vs Futuro")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    # Mapa presente
                    fig1, ax1 = plt.subplots(figsize=(6, 5))
                    cax1 = ax1.imshow(proj["base_map"], cmap="viridis", interpolation='nearest')
                    ax1.set_title("Distribuição Atual")
                    fig1.colorbar(cax1, label="Adequabilidade")
                    st.pyplot(fig1)
                
                with col2:
                    # Mapa futuro
                    fig2, ax2 = plt.subplots(figsize=(6, 5))
                    cax2 = ax2.imshow(proj["map"], cmap="viridis", interpolation='nearest')
                    ax2.set_title(f"Projeção Futura ({proj['params']['period']})")
                    fig2.colorbar(cax2, label="Adequabilidade")
                    st.pyplot(fig2)
            
            with proj_tabs[1]:
                st.write("### Mapa de Mudanças")
                
                # Cálculo da diferença
                diff_map = proj["map"] - proj["base_map"]
                
                # Mapa de diferenças
                fig, ax = plt.subplots(figsize=(10, 8))
                cax = ax.imshow(diff_map, cmap="RdBu_r", interpolation='nearest', vmin=-1, vmax=1)
                ax.set_title("Mudanças na Adequabilidade (Futuro - Presente)")
                cbar = fig.colorbar(cax, label="Mudança")
                cbar.set_ticks([-1, -0.5, 0, 0.5, 1])
                cbar.set_ticklabels(['Perda total', 'Perda parcial', 'Sem mudança', 'Ganho parcial', 'Ganho total'])
                st.pyplot(fig)
                
                # Estatísticas de mudança
                total_area = diff_map.size
                perda = np.sum(diff_map < -0.2) / total_area
                ganho = np.sum(diff_map > 0.2) / total_area
                estavel = 1 - perda - ganho
                
                st.write(f"**Proporção de área com perda de adequabilidade:** {perda:.2%}")
                st.write(f"**Proporção de área com ganho de adequabilidade:** {ganho:.2%}")
                st.write(f"**Proporção de área estável:** {estavel:.2%}")
            
            with proj_tabs[2]:
                st.write("### Estatísticas Comparativas")
                
                # Métricas de mudança de área adequada
                threshold = 0.5  # Limiar para considerar área adequada
                
                area_atual = np.mean(proj["base_map"] >= threshold)
                area_futura = np.mean(proj["map"] >= threshold)
                mudanca_relativa = (area_futura - area_atual) / area_atual * 100
                
                # Estatísticas descritivas
                stats_atual = {
                    "Área adequada (%)": area_atual * 100,
                    "Probabilidade média": np.mean(proj["base_map"]),
                    "Probabilidade máxima": np.max(proj["base_map"])
                }
                
                stats_futura = {
                    "Área adequada (%)": area_futura * 100,
                    "Probabilidade média": np.mean(proj["map"]),
                    "Probabilidade máxima": np.max(proj["map"])
                }
                
                stats_df = pd.DataFrame({
                    "Presente": stats_atual,
                    "Futuro": stats_futura,
                    "Mudança (%)": {
                        "Área adequada (%)": mudanca_relativa,
                        "Probabilidade média": (stats_futura["Probabilidade média"] / stats_atual["Probabilidade média"] - 1) * 100,
                        "Probabilidade máxima": (stats_futura["Probabilidade máxima"] / stats_atual["Probabilidade máxima"] - 1) * 100
                    }
                })
                
                st.dataframe(stats_df)
                
                # Histograma comparativo
                fig, ax = plt.subplots(figsize=(10, 6))
                sns.histplot(proj["base_map"].flatten(), bins=20, kde=True, 
                             label="Presente", alpha=0.6, ax=ax)
                sns.histplot(proj["map"].flatten(), bins=20, kde=True, 
                             label="Futuro", alpha=0.6, ax=ax)
                ax.set_xlabel("Adequabilidade")
                ax.set_ylabel("Frequência")
                ax.set_title("Comparação da Distribuição de Adequabilidade")
                ax.legend()
                st.pyplot(fig)


# =============================================================================
# Menu de Navegação (Sidebar) e Execução do App
# =============================================================================

def main():
    """
    Função principal que controla o fluxo do aplicativo e configura a barra lateral.
    """
    # Configuração da barra lateral
    st.sidebar.title("TAIPA - Navegação")
    
    # Logo (se existir)
    try:
        st.sidebar.image("logo_taipa.png", width=200)
    except:
        pass
    
    # Menu de navegação
    pagina = st.sidebar.selectbox(
        "Selecione a página", 
        ["Home", "Busca Ocorrência", "Pseudoausências", 
        "Bioclima", "Executar Modelo", "Resultados", "Projeção Futura"]
    )
    
    # Informações na barra lateral
    with st.sidebar.expander("Sobre o TAIPA"):
        st.write("""
        **TAIPA** (Tecnologia de Aprendizado Interativo em Predição Ambiental) 
        é uma plataforma educacional para modelagem de distribuição de espécies.
        
        Desenvolvido como ferramenta de ensino e pesquisa.
        """)
    
    # Exibe a página selecionada
    if pagina == "Home":
        pagina_inicial()
    elif pagina == "Busca Ocorrência":
        pagina_busca_api()
    elif pagina == "Pseudoausências":
        pagina_pseudoausencias()
    elif pagina == "Bioclima":
        pagina_variaveis_ambientais()
    elif pagina == "Executar Modelo":
        executar_modelo()
    elif pagina == "Resultados":
        resultados()
    elif pagina == "Projeção Futura":
        projecao_futura()
    
    # Rodapé
    st.sidebar.markdown("---")
    st.sidebar.markdown("<small>TAIPA v1.0.0 | 2025</small>", unsafe_allow_html=True)
    st.sidebar.markdown("<small>Desenvolvido por Pedro Higuchi</small>", unsafe_allow_html=True)

# Executa o aplicativo
if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        st.error(f"Erro não tratado: {str(e)}")
        logger.error(f"Erro não tratado na aplicação: {str(e)}", exc_info=True)