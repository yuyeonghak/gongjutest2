from io import BytesIO
import json
import re

import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st


# 원본 데이터 주소입니다.
POPULATION_URL = (
    "https://raw.githubusercontent.com/greatsong/modudata/"
    "main/data/population_yearly.csv.gz"
)
GEOJSON_URL = (
    "https://raw.githubusercontent.com/greatsong/modudata/"
    "main/data/boundaries/sigungu_kr.geojson"
)

# 고령화율 구간과 색입니다. 낮은 구간은 옅고 높은 구간은 진합니다.
AGEING_BINS = [float("-inf"), 19, 23, 28, 38, float("inf")]
AGEING_LABELS = [
    "19% 미만",
    "19% 이상 ~ 23% 미만",
    "23% 이상 ~ 28% 미만",
    "28% 이상 ~ 38% 미만",
    "38% 이상",
]
AGEING_COLORS = ["#FFF4E6", "#FDD8B3", "#F5A36C", "#D95F4B", "#7F1D1D"]


st.set_page_config(
    page_title="전국 고령화 지도",
    page_icon="🗺️",
    layout="wide",
)


def download(url: str) -> bytes:
    """인터넷에서 파일을 내려받고, 실패하면 알아보기 쉬운 오류를 냅니다."""
    response = requests.get(url, timeout=(10, 120))
    response.raise_for_status()
    return response.content


@st.cache_data(ttl=6 * 60 * 60, show_spinner=False)
def prepare_data():
    """
    최신 연도의 읍·면·동 인구를 시군구 단위로 합칩니다.

    캐시를 사용하므로 화면이 다시 그려질 때마다 큰 CSV를 재다운로드하지
    않습니다. 캐시는 6시간 뒤 새로 갱신됩니다.
    """
    population_bytes = download(POPULATION_URL)

    # '코드'는 계산용 숫자가 아니므로 처음부터 문자열로 읽습니다.
    # 전체 열 중 연도, 코드, 남녀 합계(계_) 연령 열만 읽어 메모리를 아낍니다.
    population = pd.read_csv(
        BytesIO(population_bytes),
        compression="gzip",
        dtype={"코드": "string"},
        usecols=lambda column: (
            column in {"연도", "코드"} or column.startswith("계_")
        ),
    )

    required_columns = {"연도", "코드"}
    if not required_columns.issubset(population.columns):
        raise ValueError("인구 데이터에서 '연도' 또는 '코드' 열을 찾지 못했습니다.")

    population["연도"] = pd.to_numeric(population["연도"], errors="coerce")
    if population["연도"].notna().sum() == 0:
        raise ValueError("인구 데이터에서 사용할 수 있는 연도를 찾지 못했습니다.")

    latest_year = int(population["연도"].max())
    latest = population.loc[population["연도"] == latest_year].copy()

    # 열 이름에서 실제 나이를 뽑습니다.
    # 예: '계_65세' -> 65, '계_100세 이상' -> 100
    age_by_column = {}
    for column in latest.columns:
        match = re.fullmatch(r"계_(\d+)세(?: 이상)?", str(column))
        if match:
            age_by_column[column] = int(match.group(1))

    all_age_columns = list(age_by_column)
    senior_columns = [
        column for column, age in age_by_column.items() if age >= 65
    ]
    if not all_age_columns or not senior_columns:
        raise ValueError("인구 데이터에서 연령별 '계_' 열을 찾지 못했습니다.")

    # 빈칸이나 잘못된 값은 0명으로 처리한 뒤 읍·면·동별 합계를 계산합니다.
    age_values = latest[all_age_columns].apply(
        pd.to_numeric, errors="coerce"
    ).fillna(0)

    district_population = pd.DataFrame(
        {
            # 10자리 행정동 코드의 앞 5자리가 시군구 코드입니다.
            "코드": latest["코드"].str.strip().str.slice(0, 5),
            "전체인구": age_values.sum(axis=1),
            "65세이상인구": age_values[senior_columns].sum(axis=1),
        }
    )
    district_population = (
        district_population.dropna(subset=["코드"])
        .groupby("코드", as_index=False)[["전체인구", "65세이상인구"]]
        .sum()
    )
    district_population["고령화율"] = (
        district_population["65세이상인구"]
        .div(district_population["전체인구"].where(
            district_population["전체인구"] > 0
        ))
        .mul(100)
    )

    # GeoJSON의 속성에서 지도 표시용 시도·시군구 이름을 가져옵니다.
    geojson = json.loads(download(GEOJSON_URL).decode("utf-8-sig"))
    features = geojson.get("features", [])
    if not features:
        raise ValueError("경계 데이터에 시군구 도형이 없습니다.")

    boundary_rows = []
    for feature in features:
        properties = feature.get("properties", {})
        code = str(properties.get("코드", "")).strip().zfill(5)
        properties["코드"] = code
        boundary_rows.append(
            {
                "코드": code,
                "시도": properties.get("시도", ""),
                "시군구": properties.get("시군구", ""),
            }
        )

    boundaries = pd.DataFrame(boundary_rows)
    if boundaries["코드"].duplicated().any():
        raise ValueError("경계 데이터에 중복된 시군구 코드가 있습니다.")

    # 이름이 아닌 5자리 코드로 결합해야 동명이인 지역이 섞이지 않습니다.
    map_data = boundaries.merge(
        district_population,
        on="코드",
        how="left",
        validate="one_to_one",
    )

    # right=False이므로 19.0은 두 번째 구간, 38.0은 다섯 번째 구간입니다.
    map_data["단계"] = pd.cut(
        map_data["고령화율"],
        bins=AGEING_BINS,
        labels=False,
        right=False,
    )
    map_data["고령화율표시"] = map_data["고령화율"].map(
        lambda value: f"{value:.1f}%" if pd.notna(value) else "자료 없음"
    )

    return map_data, geojson, latest_year


def discrete_colorscale(colors: list[str]) -> list[list]:
    """Plotly의 색상표를 다섯 구간에서 뚝 끊어지도록 만듭니다."""
    scale = []
    last_index = len(colors) - 1
    for index, color in enumerate(colors):
        left = 0 if index == 0 else (index - 0.5) / last_index
        right = 1 if index == last_index else (index + 0.5) / last_index
        scale.extend([[left, color], [right, color]])
    return scale


def make_map(map_data: pd.DataFrame, geojson: dict) -> go.Figure:
    """타일 배경 없이 GeoJSON 경계만 사용하는 단계구분도를 만듭니다."""
    figure = go.Figure(
        go.Choropleth(
            geojson=geojson,
            featureidkey="properties.코드",
            locations=map_data["코드"],
            z=map_data["단계"],
            zmin=0,
            zmax=4,
            colorscale=discrete_colorscale(AGEING_COLORS),
            showscale=True,
            customdata=map_data[
                ["시군구", "시도", "고령화율표시"]
            ].to_numpy(),
            hovertemplate=(
                "<b>%{customdata[0]}</b><br>"
                "시도: %{customdata[1]}<br>"
                "고령화율: %{customdata[2]}"
                "<extra></extra>"
            ),
            marker_line_color="#FFFFFF",
            marker_line_width=0.6,
            colorbar={
                "title": {"text": "고령화율 구간"},
                "tickmode": "array",
                "tickvals": [0, 1, 2, 3, 4],
                "ticktext": AGEING_LABELS,
                "thickness": 18,
                "len": 0.72,
            },
        )
    )

    # fitbounds가 대한민국 도형에 화면을 맞춥니다.
    # visible=False로 위·경도축, 바다, 기본 지도 배경을 모두 숨깁니다.
    figure.update_geos(
        fitbounds="locations",
        visible=False,
        bgcolor="rgba(0,0,0,0)",
    )
    figure.update_layout(
        height=720,
        margin={"l": 0, "r": 20, "t": 10, "b": 0},
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )
    return figure


def ranking_table(data: pd.DataFrame, highest: bool) -> pd.DataFrame:
    """표에 보여 줄 상위 또는 하위 10개 지역을 정리합니다."""
    ranked = data.dropna(subset=["고령화율"]).sort_values(
        "고령화율",
        ascending=not highest,
    ).head(10)

    table = ranked[["시도", "시군구", "고령화율"]].copy()
    table.insert(0, "순위", range(1, len(table) + 1))
    table["고령화율(%)"] = table.pop("고령화율").map(
        lambda value: f"{value:.1f}%"
    )
    return table


def main():
    st.title("전국 시군구 고령화 지도")
    st.caption("시군구별 전체 인구 중 65세 이상 인구가 차지하는 비율")

    try:
        with st.spinner("최신 인구와 지도 경계를 불러오는 중입니다..."):
            map_data, geojson, latest_year = prepare_data()
    except (requests.RequestException, ValueError, KeyError) as error:
        st.error(
            "데이터를 불러오거나 처리하지 못했습니다. "
            "잠시 뒤 새로고침해 주세요."
        )
        st.exception(error)
        st.stop()

    matched_count = int(map_data["고령화율"].notna().sum())
    st.subheader(f"{latest_year}년 기준")
    st.caption(
        f"경계 {len(map_data):,}개 중 인구 데이터가 연결된 "
        f"시군구는 {matched_count:,}개입니다."
    )

    st.plotly_chart(
        make_map(map_data, geojson),
        use_container_width=True,
        config={"displayModeBar": False},
    )

    st.divider()
    high_column, low_column = st.columns(2)
    with high_column:
        st.subheader("고령화율 높은 곳 10개")
        st.dataframe(
            ranking_table(map_data, highest=True),
            hide_index=True,
            use_container_width=True,
        )
    with low_column:
        st.subheader("고령화율 낮은 곳 10개")
        st.dataframe(
            ranking_table(map_data, highest=False),
            hide_index=True,
            use_container_width=True,
        )

    st.caption(
        "자료: 행정동 연령별 인구 및 전국 시군구 경계 "
        "(github.com/greatsong/modudata)"
    )


if __name__ == "__main__":
    main()
