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
import seaborn as sns  # Adicionado para visualização de correlação
from streamlit_folium import st_folium
from folium.plugins import Draw
from shapely.geometry import Point, Polygon
from shapely.ops import unary_union
import time
import ee
import json
from scipy.stats import gaussian_kde
import seaborn as sns
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tools.tools import add_constant
from google.oauth2 import service_account
from ee import oauth
import json


# =============================================================================
# Autenticação e Inicialização do Earth Engine
# =============================================================================



#ee.Authenticate()
#ee.Initialize(project=st.secrets["earthengine"]["project"])
def ee_initialize(force_use_service_account=False):
    if force_use_service_account or "json_data" in st.secrets:
        json_credentials = st.secrets["json_data"]
        # Se os dados já estiverem como um objeto JSON, você pode usar diretamente
        # Caso contrário, se estiverem como string, use json.loads(json_credentials)
        if isinstance(json_credentials, str):
            credentials_dict = json.loads(json_credentials)
        else:
            credentials_dict = json_credentials
        if 'client_email' not in credentials_dict:
            raise ValueError("Service account info is missing 'client_email' field.")
        credentials = service_account.Credentials.from_service_account_info(
            credentials_dict, scopes=oauth.SCOPES
        )
        ee.Initialize(credentials)
    else:
        ee.Initialize()

# # Forçar o uso da conta de serviço (útil em produção)
ee_initialize(force_use_service_account=True)

# Inicializa o Earth Engine


# =============================================================================
# Dicionários de Configuração
# =============================================================================

vis_params_dict = {
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
}

bio_descriptions_pt = {
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
    }

# =============================================================================
# Funções Auxiliares (Utilitários)
# =============================================================================

@st.cache_data(show_spinner=False)
def load_brazil_polygon():
    brazil_gdf = geobr.read_country(year=2019)
    if brazil_gdf.empty:
        st.error("Não foi possível carregar o polígono do Brasil com geobr.")
        return None
    return brazil_gdf.geometry.unary_union

brazil_polygon = load_brazil_polygon()

def load_bioclim_var(var_number):
    asset_base = st.secrets["earthengine"]["asset_base"]
    asset_path = f"{asset_base}/wc2_1_5m_bio_{var_number}"
    return ee.Image(asset_path)

@st.cache_data(show_spinner=False)
def extract_values_cached(_image, df, band_name, var_number):
    features = [ee.Feature(ee.Geometry.Point([row['longitude'], row['latitude']])) 
                for _, row in df.iterrows()]
    fc = ee.FeatureCollection(features)
    fc_extracted = _image.reduceRegions(
        collection=fc,
        reducer=ee.Reducer.first().setOutputs([band_name]),
        scale=30
    )
    dict_extracted = fc_extracted.getInfo()
    values = [f['properties'].get(band_name) 
              for f in dict_extracted['features'] if f['properties'].get(band_name) is not None]
    return values

def shapely_to_ee_geometry(shapely_geom):
    geojson = shapely_geom.__geo_interface__
    if geojson['type'] == 'Polygon':
        return ee.Geometry.Polygon(geojson['coordinates'])
    elif geojson['type'] == 'MultiPolygon':
        return ee.Geometry.MultiPolygon(geojson['coordinates'])
    else:
        st.error("Tipo de geometria não suportado para conversão.")
        return None

def add_ee_layer(self, ee_image_object, vis_params, name):
    map_id_dict = ee.Image(ee_image_object).getMapId(vis_params)
    folium.raster_layers.TileLayer(
        tiles=map_id_dict['tile_fetcher'].url_format,
        attr='Map Data &copy; Google Earth Engine',
        name=name,
        overlay=True,
        control=True
    ).add_to(self)

folium.Map.add_ee_layer = add_ee_layer

# =============================================================================
# Funções de Geração de Pseudoausências e Busca de Ocorrências
# =============================================================================

def generate_pseudo_absences_in_buffers(presence_df, n_points=100, buffer_distance=0.5):
    presence_points = [Point(lon, lat) for lon, lat in zip(presence_df["longitude"], presence_df["latitude"])]
    buffers = [point.buffer(buffer_distance) for point in presence_points]
    union_buffers = unary_union(buffers)
    allowed_region = union_buffers.intersection(brazil_polygon)
    
    if allowed_region.is_empty:
        st.error("A área permitida (buffers ∩ Brasil) está vazia. Tente diminuir o buffer.")
        return pd.DataFrame()
    
    st.write("Limites da área permitida (buffers ∩ Brasil):", allowed_region.bounds)
    st.write("Área permitida (valor em graus²):", allowed_region.area)
    
    pseudo_points = []
    attempts = 0
    max_attempts = n_points * 1000
    minx, miny, maxx, maxy = allowed_region.bounds
    while len(pseudo_points) < n_points and attempts < max_attempts:
        random_lon = np.random.uniform(minx, maxx)
        random_lat = np.random.uniform(miny, maxy)
        p = Point(random_lon, random_lat)
        if allowed_region.contains(p):
            pseudo_points.append({
                "species": "pseudo-absence",
                "latitude": random_lat,
                "longitude": random_lon
            })
        attempts += 1
    return pd.DataFrame(pseudo_points)

def fetch_gbif_occurrences(species, limit=100):
    url = "https://api.gbif.org/v1/occurrence/search"
    params = {
        "scientificName": species,
        "hasCoordinate": "true",
        "country": "BR",
        "limit": limit
    }
    response = requests.get(url, params=params)
    if response.status_code == 200:
        data = response.json()
        occurrences = []
        for record in data.get("results", []):
            lat = record.get("decimalLatitude")
            lon = record.get("decimalLongitude")
            if lat is not None and lon is not None:
                occurrences.append({
                    "species": species,
                    "latitude": lat,
                    "longitude": lon
                })
        return pd.DataFrame(occurrences)
    else:
        st.error("Erro ao buscar dados na API do GBIF")
        return pd.DataFrame()

# =============================================================================
# Funções de Visualização e Extração de Dados
# =============================================================================

def build_map(df, heatmap=False):
    """
    Cria um objeto Folium.Map com os pontos de ocorrência, plugin Draw e,
    opcionalmente, uma camada de heatmap sobreposta como FeatureGroup.
    """
    # Calcula os limites e o centro do mapa com base nos dados
    min_lat = df["latitude"].min()
    max_lat = df["latitude"].max()
    min_lon = df["longitude"].min()
    max_lon = df["longitude"].max()
    center = [(min_lat + max_lat) / 2, (min_lon + max_lon) / 2]

    m = folium.Map(location=center)
    m.fit_bounds([[min_lat, min_lon], [max_lat, max_lon]])

    # Adiciona os pontos em um FeatureGroup para que possam ser gerenciados pelo LayerControl
    points_fg = folium.FeatureGroup(name="Ocorrências", show=True)
    for _, row in df.iterrows():
        folium.CircleMarker(
            location=[row["latitude"], row["longitude"]],
            radius=3,
            color="black",
            fill=True,
            fill_color="black",
            fill_opacity=1
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
        from folium.plugins import HeatMap
        heatmap_fg = folium.FeatureGroup(name="Heatmap", show=True)
        pontos = df[["latitude", "longitude"]].values.tolist()
        # Parâmetros ajustados para manchas maiores
        HeatMap(pontos, radius=40, blur=15).add_to(heatmap_fg)
        heatmap_fg.add_to(m)

    # Adiciona o controle de layers para permitir a seleção das camadas
    folium.LayerControl().add_to(m)
    return m
    
    return m



@st.cache_data(show_spinner=True)
def get_bioclim_stats(df_occ, df_pseudo, _brazil_ee):
    """
    Extrai os valores das 19 variáveis bioclimáticas e calcula as estatísticas
    descritivas para os pontos de presença e pseudoausência.
    """
    stats_data = {}
    raw_data = {}
    stat_names = ['count', 'mean', 'std', 'min', '25%', '50%', '75%', 'max']

    for i in range(1, 20):
        bio_name = f"BIO{i}"
        bio_image = load_bioclim_var(i)
        # Obtem o nome da banda
        band_name = bio_image.bandNames().getInfo()[0]
        # Recorta a imagem para o polígono do Brasil
        bio_image_clipped = bio_image.clip(_brazil_ee)

        # Extrai os valores para os pontos de presença e pseudoausência
        presence_values = extract_values_cached(bio_image_clipped, df_occ, band_name, i)
        pseudo_values = extract_values_cached(bio_image_clipped, df_pseudo, band_name, i)
        combined_values = presence_values + pseudo_values

        # Calcula as estatísticas descritivas, ou preenche com NaN se não houver valores
        if combined_values:
            combined_stats = pd.Series(combined_values).describe()
        else:
            combined_stats = pd.Series({stat: np.nan for stat in stat_names})
        stats_data[bio_name] = combined_stats
        raw_data[bio_name] = combined_values

    return stats_data, raw_data


def plot_density(values, label, color):
    if len(values) == 0:
        return None, None
    kde = gaussian_kde(values)
    xs = np.linspace(min(values), max(values), 200)
    density = kde(xs)
    plt.plot(xs, density, label=label, color=color)
    return xs, density

def sample_bio_value(image, lat, lon, band_name, scale=30):
    pt = ee.Geometry.Point([lon, lat])
    sample = image.sample(pt, scale).first()
    return sample.get(band_name).getInfo()


# =============================================================================
# Páginas da Aplicação (Interface com o Usuário)
# =============================================================================

def home():
    st.title("TAIPA - Tecnologia de Aprendizado Interativo em Predição Ambiental")
    st.write("""
    **Bem-vindo à TAIPA!**

    A TAIPA é uma plataforma interativa para o ensino de Modelagem de Distribuição de Espécies (SDM)
    utilizando o algoritmo MaxEnt. Aqui, você pode explorar dados de ocorrência, gerar pseudoausências,
    visualizar variáveis ambientais e executar modelos simulados.
    """)
    st.info("Utilize o menu lateral para navegar pelas funcionalidades.")

def search_api():
    st.title("Busca de Ocorrências via API do GBIF")
    st.warning("Os dados da API do GBIF são gratuitos, mas requerem citação. Confira: https://www.gbif.org/citation-guidelines")
    
    # Formulário de busca
    with st.form(key="search_form"):
        species = st.text_input("Digite o nome científico da espécie:")
        submitted = st.form_submit_button("Buscar Ocorrências")
        if submitted and species.strip() != "":
            st.info("Buscando dados na API do GBIF...")
            df_api = fetch_gbif_occurrences(species, limit=100)
            st.session_state.df_api = df_api
            st.session_state.species = species
            # Limpa a flag do heatmap ao buscar novos dados
            st.session_state.heatmap_generated = False

    if "df_api" in st.session_state:
        df_api = st.session_state.df_api
        st.write("Total de ocorrências retornadas:", len(df_api))
        if not df_api.empty:
            st.write("Visualização dos dados obtidos:")
            st.write(df_api.head())
            
            # Botão para ativar o heatmap (a flag é mantida no session_state)
            if st.button("Gerar Heatmap de Ocorrência"):
                st.session_state.heatmap_generated = True

            # Constrói o mapa base (com ou sem heatmap, conforme flag)
            m = build_map(df_api, heatmap=st.session_state.get("heatmap_generated", False))
            
            # Renderiza o mapa e captura os dados interativos
            map_data = st_folium(m, width=700, height=500)
            
            # Verifica se o usuário desenhou algum polígono para remoção de pontos
            if map_data.get("all_drawings"):
                polygon_features = [
                    feature for feature in map_data["all_drawings"]
                    if feature.get("geometry", {}).get("type") == "Polygon"
                ]
                if polygon_features:
                    st.info("Polígono(s) desenhado(s) detectado(s).")
                    if st.button("Remover pontos dentro do polígono"):
                        poly_coords = polygon_features[0]["geometry"]["coordinates"][0]
                        polygon_shapely = Polygon(poly_coords)
                        indices_to_remove = []
                        for idx, row in df_api.iterrows():
                            point = Point(row["longitude"], row["latitude"])
                            if polygon_shapely.contains(point):
                                indices_to_remove.append(idx)
                        if indices_to_remove:
                            st.session_state.df_api = df_api.drop(indices_to_remove).reset_index(drop=True)
                            st.success(f"{len(indices_to_remove)} ponto(s) removido(s) dentro do polígono.")
                            # Recria o mapa atualizado com os pontos restantes
                            df_api = st.session_state.df_api
                            m = build_map(df_api, heatmap=st.session_state.get("heatmap_generated", False))
                            st_folium(m, width=700, height=500)
                        else:
                            st.info("Nenhum ponto encontrado dentro do polígono.")
        else:
            st.warning("Nenhum dado encontrado para a espécie informada.")

def pseudo_absences_page():
    st.title("Geração de Pseudoausências")
    st.write("Gera pontos de pseudoausência utilizando buffers dos pontos de presença (limitado ao Brasil).")
    if "df_api" in st.session_state:
        df_presence = st.session_state.df_api
        st.write("Visualização dos dados de presença:", df_presence.head())
        n_points = st.slider("Número de pseudoausências a gerar", min_value=50, max_value=500, value=100, step=10)
        # Slider para buffer em Km
        buffer_distance_km = st.slider("Tamanho do buffer (em Km)", min_value=1, max_value=200, value=50, step=1)
        # Converte de Km para graus (1 grau ≈ 111 Km)
        buffer_distance_degrees = buffer_distance_km / 111.0
        
        if st.button("Gerar Pseudoausências"):
            pseudo_df = generate_pseudo_absences_in_buffers(df_presence, n_points, buffer_distance_degrees)
            st.session_state.df_pseudo = pseudo_df
            st.success(f"{len(pseudo_df)} pontos de pseudoausência gerados (Dentro dos buffers).")
        if "df_pseudo" in st.session_state:
            st.write("Visualização das pseudoausências:", st.session_state.df_pseudo.head())
            min_lat = df_presence["latitude"].min()
            max_lat = df_presence["latitude"].max()
            min_lon = df_presence["longitude"].min()
            max_lon = df_presence["longitude"].max()
            center = [(min_lat + max_lat) / 2, (min_lon + max_lon) / 2]
            m = folium.Map(location=center)
            m.fit_bounds([[min_lat, min_lon], [max_lat, max_lon]])
            for _, row in df_presence.iterrows():
                folium.CircleMarker(
                    location=[row["latitude"], row["longitude"]],
                    radius=3,
                    color="blue",
                    fill=True,
                    fill_color="blue",
                    fill_opacity=1
                ).add_to(m)
            for _, row in st.session_state.df_pseudo.iterrows():
                folium.CircleMarker(
                    location=[row["latitude"], row["longitude"]],
                    radius=3,
                    color="red",
                    fill=True,
                    fill_color="red",
                    fill_opacity=1
                ).add_to(m)
            st_folium(m, width=700, height=500)
    else:
        st.warning("Dados de presença não encontrados. Execute a busca via API primeiro.")


def environmental_variables_all():
    st.title("Estatísticas Descritivas - Bioclima")

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

    brazil_ee = shapely_to_ee_geometry(brazil_polygon)
    if brazil_ee is None:
        st.error("Não foi possível converter o polígono do Brasil.")
        return

    # Usa uma função cacheada para extrair os dados das variáveis bioclimáticas
    with st.spinner("Extraindo valores das variáveis bioclimáticas..."):
        stats_data, raw_data = get_bioclim_stats(df_occ, df_pseudo, brazil_ee)

    # Exibe as estatísticas descritivas dos dados combinados
    df_stats = pd.DataFrame(stats_data).T
    st.subheader("Estatísticas Descritivas")
    st.dataframe(df_stats)

    # Cria DataFrame com os valores extraídos para correlação e VIF
    raw_data_df = pd.DataFrame({k: pd.Series(v) for k, v in raw_data.items()})
    corr_matrix = raw_data_df.corr()

    st.subheader("Matriz de Correlação Pairwise")
    st.dataframe(corr_matrix)
    
    # Exibe o heatmap da matriz de correlação
    fig, ax = plt.subplots()
    sns.heatmap(corr_matrix, annot=False, cmap="coolwarm", square=True, ax=ax)
    ax.set_title("Matriz de Correlação Pairwise")
    st.pyplot(fig)
    
    # Cálculo do VIF incluindo o intercepto (constante)
    with st.spinner("Calculando VIF para todas as variáveis (com intercepto)..."):
        raw_data_df_clean = raw_data_df.dropna()
        if raw_data_df_clean.empty:
            st.warning("Dados insuficientes para cálculo do VIF após remoção de NaNs.")
            return
        # Adiciona a constante
        X = add_constant(raw_data_df_clean)
        vif_all = pd.DataFrame()
        vif_all["Feature"] = X.columns
        vif_all["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
        st.subheader("VIF - Valores Iniciais para Todas as Variáveis (com intercepto)")
        st.dataframe(vif_all)
    
    # Eliminação stepwise: remoção das variáveis com maior VIF (excluindo a constante)
    with st.spinner("Calculando VIF e realizando eliminação stepwise..."):
        threshold = 10  # Limiar desejado
        variables = list(raw_data_df_clean.columns)
        while True:
            # Adiciona constante para o cálculo
            X_temp = add_constant(raw_data_df_clean[variables])
            vif_df = pd.DataFrame()
            vif_df["Feature"] = X_temp.columns
            vif_df["VIF"] = [variance_inflation_factor(X_temp.values, i) for i in range(X_temp.shape[1])]
            # Exclui a constante da análise para remoção
            vif_df_no_const = vif_df[vif_df["Feature"] != "const"]
            max_vif = vif_df_no_const["VIF"].max()
            if max_vif < threshold or len(variables) == 1:
                break
            max_var = vif_df_no_const.sort_values("VIF", ascending=False)["Feature"].iloc[0]
            variables.remove(max_var)
        
        X_final = add_constant(raw_data_df_clean[variables])
        final_vif = pd.DataFrame()
        final_vif["Feature"] = X_final.columns
        final_vif["VIF"] = [variance_inflation_factor(X_final.values, i) for i in range(X_final.shape[1])]
    
    st.subheader("VIF Final Após Eliminação Stepwise")
    final_vif = final_vif[final_vif["Feature"] != "const"]
    st.dataframe(final_vif)
    st.info(f"Variáveis selecionadas para modelagem (VIF < {threshold}): {', '.join(variables)}")

    # Permite que o usuário selecione as variáveis para visualização
    selected_vars = st.multiselect(
        "Selecione as variáveis para visualizar no mapa:",
        options=variables,
        default=variables
    )

    if selected_vars:
        # Cria um mapa centrado no polígono do Brasil
        bounds = brazil_polygon.bounds
        m = folium.Map(tiles="OpenStreetMap")
        m.fit_bounds([[bounds[1], bounds[0]], [bounds[3], bounds[2]]])
    
        # Para cada variável selecionada, adiciona uma camada do Earth Engine
        for var in selected_vars:
            # Extrai o número da variável (por exemplo, de "BIO3" extrai 3)
            var_number = int(var.replace("BIO", ""))
            bio_image = load_bioclim_var(var_number)
            band_name = bio_image.bandNames().getInfo()[0]
            # Recorta a imagem para o polígono do Brasil
            bio_image_clipped = bio_image.clip(shapely_to_ee_geometry(brazil_polygon))
            # Utiliza os parâmetros de visualização, se disponíveis
            vis_params = vis_params_dict.get(var_number, {})
            # Adiciona a camada ao mapa (o método add_ee_layer já foi incorporado à classe do folium.Map)
            m.add_ee_layer(bio_image_clipped, vis_params, f"{var} - {bio_descriptions_pt[var_number]}")
    
        # Adiciona os pontos de presença (preto)
        for _, row in df_occ.iterrows():
            folium.CircleMarker(
                location=[row["latitude"], row["longitude"]],
                radius=3,
                color="black",
                fill=True,
                fill_color="black",
                fill_opacity=1
            ).add_to(m)
    
        # Adiciona os pontos de pseudoausência (vermelho)
        for _, row in df_pseudo.iterrows():
            folium.CircleMarker(
                location=[row["latitude"], row["longitude"]],
                radius=3,
                color="red",
                fill=True,
                fill_color="red",
                fill_opacity=1
            ).add_to(m)
    
        # Adiciona um controle de camadas para que o usuário possa ativar/desativar as variáveis
        folium.LayerControl().add_to(m)
    
        st.subheader("Visualização das Variáveis Selecionadas no Brasil")
        st_folium(m, width=700, height=500)

def run_model():
    st.title("Execução do Modelo MaxEnt")
    if st.button("Executar Modelo"):
        st.info("Processando o modelo...")
        time.sleep(2)
        st.success("Modelo executado com sucesso!")
        simulated_map = np.random.rand(100, 100)
        fig, ax = plt.subplots()
        cax = ax.imshow(simulated_map, cmap="viridis")
        ax.set_title("Mapa de Probabilidade Simulado")
        fig.colorbar(cax)
        st.pyplot(fig)
        st.session_state["model_results"] = simulated_map

def results():
    st.title("Resultados do Modelo")
    if "model_results" in st.session_state:
        simulated_map = st.session_state["model_results"]
        st.write("### Mapa de Probabilidade")
        fig, ax = plt.subplots()
        cax = ax.imshow(simulated_map, cmap="viridis")
        ax.set_title("Mapa de Probabilidade")
        fig.colorbar(cax)
        st.pyplot(fig)
        st.write("### Curva de Resposta Simulada")
        x = np.linspace(0, 1, 100)
        response = np.exp(-((x - 0.5) ** 2) / (2 * 0.1 ** 2))
        fig2, ax2 = plt.subplots()
        ax2.plot(x, response)
        ax2.set_xlabel("Valor da variável")
        ax2.set_ylabel("Resposta")
        ax2.set_title("Curva de Resposta Simulada")
        st.pyplot(fig2)
    else:
        st.warning("Nenhum resultado encontrado. Execute o modelo na seção 'Executar Modelo'.")

def future_projection():
    st.title("Projeção Futura")
    scenario = st.selectbox("Cenário Climático Futuro", ["2050", "2070"])
    if st.button("Executar Projeção"):
        st.info(f"Processando projeção para o cenário {scenario}...")
        time.sleep(2)
        st.success("Projeção realizada com sucesso!")
        simulated_future_map = np.random.rand(100, 100)
        fig, ax = plt.subplots()
        cax = ax.imshow(simulated_future_map, cmap="plasma")
        ax.set_title(f"Mapa de Projeção para {scenario}")
        fig.colorbar(cax)
        st.pyplot(fig)

# =============================================================================
# Menu de Navegação (Sidebar) e Execução do App
# =============================================================================

st.sidebar.title("TAIPA - Navegação")
page = st.sidebar.selectbox("Sfelecione a página", 
    ["Home", "Busca Ocorrência",  "Pseudoausências", 
     "Bioclima", "Executar Modelo", "Resultados", "Projeção Futura"])


if page == "Home":
    home()
elif page == "Busca Ocorrência":
    search_api()
elif page == "Pseudoausências":
    pseudo_absences_page()
elif page == "Bioclima":
    environmental_variables_all()
elif page == "Executar Modelo":
    st.warning("Em desenvolvimento...")
elif page == "Resultados":
    st.warning("Em desenvolvimento...")
elif page == "Projeção Futura":
    st.warning("Em desenvolvimento...")
