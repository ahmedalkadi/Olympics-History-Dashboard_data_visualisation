from dash import html, dcc
from dash.dependencies import Input, Output
import pandas as pd
import plotly.express as px
import numpy as np
import base64
from olympics_predictions import *
import dash_bootstrap_components as dbc


# Sample DataFrame for demonstration
df = pd.read_csv("csv_data/olympics.csv")

# Calculate the metrics
def total_athletes():
    return len(df['Name'].unique())

def total_events():
    return len(df['Event'].unique())

def total_medals():
    return df['Medal'].count()

def total_countries_participated():
    return len(df['NOC'].unique())

def average_age():
    return np.round(df['Age'].mean(), 2)


def overall_participation():
    participation_trends = df.groupby(['Year', 'Season'])['ID'].nunique().reset_index()

    fig = px.line(participation_trends, x='Year', y='ID', color='Season')
    return dcc.Graph(figure=fig, id='partic')

def age_dist():
    fig = px.histogram(df, x='Age', nbins=30)
    return dcc.Graph(figure=fig, id='age_dist')

def medals_by_sport():
    sport_medals = df.groupby(['Sport', 'Medal']).size().unstack(fill_value=0)
    color_mapping = {'Gold': 'goldenrod', 'Silver': 'silver', 'Bronze': 'saddlebrown'}
    fig = px.bar(sport_medals, x=sport_medals.index, y=['Gold', 'Silver', 'Bronze'], color_discrete_map=color_mapping)
    return dcc.Graph(figure=fig, id='medals_sport')


def sports_evo():
    sports_evolution = df.groupby(['Year', 'Sport']).size().reset_index(name='Count')
    fig = px.line(sports_evolution, x='Year', y='Count', color='Sport')
    return dcc.Graph(figure=fig, id='sport_evo')

def gender_participation():
    gender_participation = df.groupby(['Year', 'Sex']).size().reset_index(name='Count')
    color_mapping = {'M': '#82B2DA', 'F': 'hotpink'} 
    fig = px.line(gender_participation, x='Year', y='Count', color='Sex', color_discrete_map=color_mapping)
    return dcc.Graph(figure=fig, id='gender_partic')

def generate_top_nations_performance_chart(df):
    nation_medals_over_time = df.groupby(['Year', 'NOC', 'Medal']).size().unstack(fill_value=0)
    nation_medals_total = nation_medals_over_time.groupby('NOC').sum().sum(axis=1)
    top_nations = nation_medals_total.nlargest(2).index
    nation_medals_top2 = nation_medals_over_time[nation_medals_over_time.index.get_level_values('NOC').isin(top_nations)]
    nation_medals_top2 = nation_medals_top2.drop(columns=['No Medal'], errors='ignore')
    nation_medals_top2 = nation_medals_top2.reset_index()
    nation_medals_melted = nation_medals_top2.melt(id_vars=['Year', 'NOC'], var_name='Medal', value_name='Count')
    color_mapping = {'Gold': 'goldenrod', 'Silver': 'grey', 'Bronze': 'saddlebrown'}
    fig = px.line(nation_medals_melted, x='Year', y='Count', color='Medal',
                facet_col='NOC',
                facet_col_wrap=2, color_discrete_map=color_mapping)
    return dcc.Graph(figure=fig)

def top_countries_by_medal_counts(df):
    medal_counts = df.groupby(['region', 'Medal']).size().unstack(fill_value=0)
    top_countries = medal_counts.sum(axis=1).sort_values(ascending=False).head(20)
    top_medal_counts = medal_counts.loc[top_countries.index]
    color_mapping = {'Gold': 'goldenrod', 'Silver': 'grey', 'Bronze': 'saddlebrown'}

    fig = px.bar(top_medal_counts,
                x=top_medal_counts.index,
                y=['Gold', 'Silver', 'Bronze'],
                color_discrete_map=color_mapping)

    fig.update_layout(xaxis_categoryorder='total descending')
    return dcc.Graph(figure=fig, id='bar-plot-countries')

def top_sports_by_medal_counts(df):
    sport_medals = df.groupby(['Sport', 'Medal']).size().unstack(fill_value=0)
    top_sports = sport_medals.sum(axis=1).sort_values(ascending=False).head(20)
    top_sport_medals = sport_medals.loc[top_sports.index]
    color_mapping = {'Gold': 'goldenrod', 'Silver': 'grey', 'Bronze': 'saddlebrown'}
    fig = px.bar(top_sport_medals,
                x=top_sport_medals.index,
                y=['Gold', 'Silver', 'Bronze'],
                color_discrete_map=color_mapping)
    fig.update_layout(xaxis_categoryorder='total descending')

    return dcc.Graph(figure=fig, id='bar-plot-sports')

def top_sports_by_gender_medals(df):
    sport_medals = df.groupby(['Sport', 'Medal', 'Sex']).size().unstack(fill_value=0)
    top_sports = sport_medals.sum(axis=1).nlargest(20).index
    top_sport_medals = sport_medals.loc[top_sports]
    top_sport_medals.reset_index(inplace=True)
    color_mapping = {'F': 'hotpink', 'M': '#82B2DA'}
    fig = px.bar(top_sport_medals,
                x='Sport',
                y=['M', 'F'],
                color_discrete_map=color_mapping)
    fig.update_layout(xaxis_categoryorder='total descending')

    return dcc.Graph(figure=fig, id='sport-medals')

def gender_distribution_pie_chart(df):
    gender_counts = df.groupby('Sex')['ID'].count().reset_index(name='Count')
    
    color_mapping = {'M': '#82B2DA', 'F': 'hotpink'}
    
    fig = px.pie(gender_counts, values='Count', names='Sex')
    fig.update_traces(marker=dict(colors=[color_mapping[gender] for gender in gender_counts['Sex']]))

    return dcc.Graph(figure=fig, id='gender-pie-chart')


def team_vs_individual_sports(df):
    team_sports = ['Basketball', 'Football', 'Hockey', 'Volleyball']
    df['Team/Individual'] = df['Sport'].apply(lambda x: 'Team Sports' if x in team_sports else 'Individual Sports')
    team_vs_individual = df.groupby(['Year', 'Team/Individual']).size().reset_index(name='Count')
    fig = px.bar(team_vs_individual, x='Year', y='Count', color='Team/Individual',
                barmode='group')
    
    return dcc.Graph(figure=fig, id='team-vs-individual')

def success_rate_by_sport(df):
    top_nations = df['NOC'].value_counts().head(5).index
    df_top5 = df[df['NOC'].isin(top_nations)]
    success_rate = df_top5.groupby(['Sport', 'NOC']).apply(lambda x: x['Medal'].count() / x['ID'].nunique()).reset_index(name='Success Rate')
    top_12_sports = success_rate.groupby('Sport')['Success Rate'].mean().nlargest(12).index
    success_rate_top12 = success_rate[success_rate['Sport'].isin(top_12_sports)]
    
    fig = px.bar(success_rate_top12, x='Sport', y='Success Rate', color='NOC',
                labels={'Success Rate': 'Success Rate (%)', 'Sport': 'Sport'})
    
    return dcc.Graph(figure=fig, id='success-rate-graph')

def athlete_biometrics_histogram(df, selected_sport=None):
    if selected_sport:
        filtered_df = df[df['Sport'] == selected_sport]
        title_suffix = f' for {selected_sport}'
    else:
        filtered_df = df
        title_suffix = ''
    
    fig_combined = px.histogram(filtered_df, x=['Height', 'Weight'])
    
    return dcc.Graph(figure=fig_combined, id='athlete-biometrics-histogram')


def gender_distribution_by_sport(df):

    sport_gender = df[['region', 'Sex', 'ID']].groupby(['region', 'Sex']).count().reset_index()
    sport_gender = sport_gender.rename(columns={"ID": "Count"})
    
    sport_gender_pivot = sport_gender.pivot(index='region', columns='Sex', values='Count').reset_index()
    sport_gender_pivot.columns.name = None
    sport_gender_pivot = sport_gender_pivot.rename(columns={"M": "Male", "F": "Female"})
    sport_gender_pivot = sport_gender_pivot.reset_index(drop=True)
    sport_gender_pivot = sport_gender_pivot.fillna(0)

    sport_gender_pivot["count"] = sport_gender_pivot["Male"] + sport_gender_pivot["Female"]
    sport_gender_pivot = sport_gender_pivot.sort_values(by="count", ascending=False)
    
    color_mapping = {'Female': 'hotpink', 'Male': '#82B2DA'}
    
    fig = px.bar(sport_gender_pivot.head(20),
                x='region',
                y=['Male', 'Female'],
                color_discrete_map=color_mapping)
    fig.update_layout(xaxis_categoryorder='total descending')

    return dcc.Graph(figure=fig)

def generate_team_vs_individual_graph(df):
    team_sports = ['Basketball', 'Football', 'Hockey', 'Volleyball']
    df['Team/Individual'] = df['Sport'].apply(lambda x: 'Team Sports' if x in team_sports else 'Individual Sports')
    team_vs_individual = df.groupby(['Year', 'Season', 'Team/Individual']).size().reset_index(name='Count')
    valid_years = df['Year'].unique()
    team_vs_individual_filtered = team_vs_individual[team_vs_individual['Year'].isin(valid_years)]
    fig = px.bar(team_vs_individual_filtered, x='Year', y='Count', color='Team/Individual', facet_col='Season',
                barmode='group')
    return dcc.Graph(figure=fig, id='team-vs-individual-graph')

def world_partic():
    participants_by_country = df.groupby('NOC')['Name'].nunique().reset_index(name='Participant Count')

    # Load the built-in world map data provided by Plotly Express
    world_map = px.data.gapminder()

    # Merge the world map data with the participants data based on country codes (ISO Alpha-3)
    map_with_participants = pd.merge(world_map, participants_by_country, how='left', left_on='iso_alpha', right_on='NOC')

    # Plot the choropleth map with custom colors
    fig = px.choropleth(map_with_participants, 
                        locations='iso_alpha', 
                        color='Participant Count', 
                        hover_name='country',
                        color_continuous_scale=px.colors.sequential.Viridis)

    return dcc.Graph(figure=fig, id='world-map')

def sports_evo():
    sport_evo=df[["Sport", "Year","ID"]].groupby(["Sport","Year"]).count().sort_values(["Year","ID"], ascending=False).reset_index()
    sport_evo=sport_evo.rename(columns={"ID": "Count"})
    sport_evo = sport_evo[sport_evo['Sport'].isin(sport_evo.Sport.head(5))]
    fig=px.line(sport_evo, x="Year", y="Count", title="Trends in Performance Across Olympic Editions" ,color="Sport")
    return dcc.Graph(figure=fig, id='sports_evo')

app = dash.Dash(external_stylesheets=[dbc.themes.SIMPLEX], suppress_callback_exceptions=True)
server= app.server

# Load the built-in world map data provided by Plotly Express
world_map = px.data.gapminder()

def image_source(img):
    image_filename = f'assets/{img}'
    encoded_image = base64.b64encode(open(image_filename, 'rb').read())
    src='data:image/png;base64,{}'.format(encoded_image.decode())
    return src


navbar = dbc.Navbar(
    [
        dbc.NavbarBrand(
            html.A(
                dbc.Row(
                    [
                        dbc.Col(
                            html.Img(src=image_source("olympics.png"), height="60px", style={"margin-left": "70px"}),
                            style={"margin-right": "5px"},
                        ),
                        dbc.Col(html.H1("Olympics Dashboard", className="ml-2 align-self-center", style={"font-size": "20px", "text-decoration": "none", "color": "#2D3C6B", "text-decoration": "none", "font-weight": "bold"})),
                    ],
                    align="center",
                ),
                style={"text-decoration": "none"}
            )
        ),
        dbc.Nav(
            [
                dbc.NavItem(dbc.NavLink(html.Img(src=image_source("home.png"), height="40px"), href="/", style={"padding-left": "20px"})),
                dbc.NavItem(dbc.NavLink(html.Img(src=image_source("pred.png"), height="40px"), href="/pred", style={"padding-left": "20px"})),
            ],
            className="mr-auto",
            navbar=True,
            style = {
                "padding-left": "950px"
            }
        ),
    ],
    color="light",
    dark=False,
    className="shadow-sm mb-5 bg-white",
    sticky="top",
    style={
        "height": "80px",
        "width": "100%", 
    }
)





big_numbers = dbc.Row(
    [
        dbc.Col(
            dbc.Card(
                dbc.CardBody(
                    [
                        html.Img(src="/assets/athlete.png", className="img-fluid", style={'width': '80px', 'height': '80px'}),
                        html.H4(id="athlete-count", style={'color': 'black', 'font-weight': 'bold', 'padding-top': '20px'}),
                    ]
                ),
                className="shadow p-3 mb-5 bg-white rounded h-80 d-flex justify-content-center align-items-center",
            ),
            className="col"
        ),
        dbc.Col(
            dbc.Card(
                dbc.CardBody(
                    [
                        html.Img(src="/assets/sports.png", className="img-fluid", style={'width': '80px', 'height': '80px'}),
                        html.H4(id="event-count", style={'color': 'black', 'font-weight': 'bold', 'padding-top': '20px', 'padding-left': '15px'}),
                    ]
                ),
                className="shadow p-3 mb-5 bg-white rounded h-80 d-flex justify-content-center align-items-center",
            ),
            className="col"
        ),
        dbc.Col(
            dbc.Card(
                dbc.CardBody(
                    [
                        html.Img(src="/assets/medal.png", className="img-fluid", style={'width': '80px', 'height': '80px'}),
                        html.H4(id="medal-count", style={'color': 'black', 'font-weight': 'bold', 'padding-top': '20px', 'padding-right': '5px'}),
                    ]
                ),
                className="shadow p-3 mb-5 bg-white rounded h-80 d-flex justify-content-center align-items-center",
            ),
            className="col"
        ),
        dbc.Col(
            dbc.Card(
                dbc.CardBody(
                    [
                        html.Img(src="/assets/countries.png", className="img-fluid", style={'width': '80px', 'height': '80px'}),
                        html.H4(id="country-count", style={'color': 'black', 'font-weight': 'bold', 'padding-top': '20px', 'padding-left':'18px'}),
                    ]
                ),
                className="shadow p-3 mb-5 bg-white rounded h-80 d-flex justify-content-center align-items-center",
            ),
            className="col"
        ),
        dbc.Col(
            dbc.Card(
                dbc.CardBody(
                    [
                        html.Img(src="/assets/age.png", className="img-fluid", style={'width': '80px', 'height': '80px'}),
                        html.H4(id="avg-age", style={'color': 'black', 'font-weight': 'bold', 'padding-top': '19px', 'padding-left':'8px'}),
                    ]
                ),
                className="shadow p-3 mb-5 bg-white rounded h-80 d-flex justify-content-center align-items-center",
            ),
            className="col"
        ),
    ],
    className="row",
)

world_map_card = dbc.Card(
    dbc.CardBody(
                    [
                        html.H4("Geographical Distribution of Participants", className="card-title"),
                        world_partic()
                    ]
                ),
    className="shadow p-3 mb-5 bg-white rounded" )


gender_participation_card = dbc.Card(
    dbc.CardBody(
        [
            html.H4("Gender Participation", className="card-title"),
            gender_participation(),
        ]
    ),
    className="shadow p-3 mb-5 bg-white rounded",
)


gender_medals_card = dbc.Card(
    dbc.CardBody(
        [
            html.H4("Top Nations Performance Over Time", className="card-title"),
            generate_top_nations_performance_chart(df),
        ]
    ),
    className="shadow p-3 mb-5 bg-white rounded",
)


gender_analysis = dbc.Row([
    dbc.Col(gender_participation_card, width=6),
    dbc.Col(gender_medals_card, width=6)
])

overall_participation_card = dbc.Card(
    dbc.CardBody(
        [
            html.H4("Overall Participation Trends", className="card-title"),
            overall_participation()
        ]
    ),
    className="shadow p-3 mb-5 bg-white rounded"
)

age_distribution_card = dbc.Card(
    dbc.CardBody(
        [
            html.H4("Age Distribution of Athletes", className="card-title"),
            age_dist()
        ]
    ),
    className="shadow p-3 mb-5 bg-white rounded"
)

medals_by_sport_card = dbc.Card(
    dbc.CardBody(
        [
            html.H4("Medals by Sport", className="card-title"),
            medals_by_sport()
        ]
    ),
    className="shadow p-3 mb-5 bg-white rounded"
)

sports_evolution_card = dbc.Card(
    dbc.CardBody(
        [
            html.H4("Evolution of Olympic Sports", className="card-title"),
            sports_evo()
        ]
    ),
    className="shadow p-3 mb-5 bg-white rounded"
)

country_medal_count_card = dbc.Card(
    dbc.CardBody(
        [
            html.H4("Top Countries by Medal Counts", className="card-title"),
            top_countries_by_medal_counts(df)
        ]
    ),
    className="shadow p-3 mb-5 bg-white rounded"
)

sport_medal_count_card = dbc.Card(
    dbc.CardBody(
        [
            html.H4("Top Sports by Medal Counts", className="card-title"),
            top_sports_by_medal_counts(df)
        ]
    ),
    className="shadow p-3 mb-5 bg-white rounded"
)

sport_medal_count_gender_card = dbc.Card(
    dbc.CardBody(
        [
            html.H4("Top Sports by Gender Medals", className="card-title"),
            top_sports_by_gender_medals(df)
        ]
    ),
    className="shadow p-3 mb-5 bg-white rounded"
)

gender_pie_chart_card = dbc.Card(
    dbc.CardBody(
        [
            html.H4("Distribution of Medals by Gender", className="card-title"),
            gender_distribution_pie_chart(df)
        ]
    ),
    className="shadow p-3 mb-5 bg-white rounded"
)

team_individual_sports_card = dbc.Card(
    dbc.CardBody(
        [
            html.H4("Team vs. Individual Sports Participation", className="card-title"),
            team_vs_individual_sports(df)
        ]
    ),
    className="shadow p-3 mb-5 bg-white rounded"
)

success_rate_by_sport_card = dbc.Card(
    dbc.CardBody(
        [
            html.H4("Success Rate by Sport and Country", className="card-title"),
            success_rate_by_sport(df)
        ]
    ),
    className="shadow p-3 mb-5 bg-white rounded"
)

sports_sorted = sorted(df['Sport'].unique())

athlete_biometrics_histogram_card = dbc.Card(
    dbc.CardBody(
        [
            html.H4("Distribution of Athlete Biometrics", className="card-title"),
            html.Div(id='athlete-biometrics-histogram-container'),
            dcc.Dropdown(
                id='sport-dropdown',
                options=[{'label': sport, 'value': sport} for sport in sports_sorted],
                value=None,
                placeholder='Select a sport',
                clearable=True
            )
        ]
    ),
    className="shadow p-3 mb-5 bg-white rounded"
)

gender_distribution_by_sport_card = dbc.Card(
    dbc.CardBody(
        [
            html.H4("Gender Distribution by Sport", className="card-title"),
            gender_distribution_by_sport(df)
        ]
    ),
    className="shadow p-3 mb-5 bg-white rounded"
)

team_vs_individual_card = dbc.Card(
    dbc.CardBody(
        [
            html.H4("Team vs. Individual Sports Participation", className="card-title"),
            generate_team_vs_individual_graph(df)
        ]
    ),
    className="shadow p-3 mb-5 bg-white rounded"
)


pred_layout = create_layout()



home_layout = html.Div(
    [
        navbar,
        dbc.Container(
            [
                html.Br(),
                big_numbers,
                html.Br(), 
                world_map_card,
                html.Br(),
                dbc.Row([
                    dbc.Col(sports_evolution_card, width=12),
                ]),
                html.Br(),
                dbc.Row([
                    dbc.Col(team_vs_individual_card, width=12),
                ]),
                html.Br(),
                dbc.Row([
                    dbc.Col(athlete_biometrics_histogram_card, width=12),
                ]),
                html.Br(),
                dbc.Row([
                    dbc.Col(overall_participation_card, width=6),
                    dbc.Col(age_distribution_card, width=6)
                ]),
                html.Br(),
                dbc.Row([
                    dbc.Col(gender_medals_card, width=12),
                ]),
                html.Br(),
                dbc.Row([
                    dbc.Col(medals_by_sport_card, width=12)
                ]),
                html.Br(),
                dbc.Row([
                    dbc.Col(country_medal_count_card, width=6),
                    dbc.Col(sport_medal_count_card, width=6)
                ]),
                html.Br(),
                dbc.Row([
                    dbc.Col(sport_medal_count_gender_card, width=6),
                    dbc.Col(gender_pie_chart_card, width=6)
                ]),          
                html.Br(),
                dbc.Row([
                    dbc.Col(gender_distribution_by_sport_card, width=6),
                    dbc.Col(gender_participation_card, width=6),
                ])
            ],
            fluid=True,
        ),
    ]
)

create_callbacks(app)

@app.callback(
    [Output("athlete-count", "children"),
     Output("event-count", "children"),
     Output("medal-count", "children"),
     Output("country-count", "children"),
     Output("avg-age", "children")],
    [Input("world-map", "clickData")]
)

def update_numbers(clickData):
    if clickData:
        # Get the clicked country code
        country_code = clickData['points'][0]['location']

        # Filter data for the clicked country
        country_data = df[df['NOC'] == country_code]

        # Update the big number boxes with country-specific data
        return (
            len(country_data['Name'].unique()),
            len(country_data['Event'].unique()),
            country_data['Medal'].count(),
            1,  # Number of countries (since we are showing data for one country)
            country_data['Age'].mean()
        )
    else:
        # If no country is clicked, show overall data
        return (
            total_athletes(),
            total_events(),
            total_medals(),
            total_countries_participated(),
            average_age()
        )

@app.callback(
    Output('athlete-biometrics-histogram-container', 'children'),
    [Input('sport-dropdown', 'value')]
)
def update_biometrics_histogram(selected_sport):
    return athlete_biometrics_histogram(df, selected_sport)

@app.callback(
    Output('page-content', 'children'),
    [Input('url', 'pathname')]
)
def display_page(pathname):
    if pathname == '/pred':
        return pred_layout
    else:
        return home_layout

# Define the app layout
app.layout = html.Div([
    dcc.Location(id='url', refresh=False),
    html.Div(id='page-content')
])
        

if __name__ == '__main__':
    app.run_server(debug=True)