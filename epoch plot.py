# -*- coding: utf-8 -*-
"""
Created on Sun Oct  8 11:43:48 2023

@author: panay
"""
import pandas as pd
import plotly.express as px
import plotly.io as pio

df = pd.read_csv('./kl divergence rpvael 150 cifar10.csv',encoding='utf7')
df_plot = df.groupby('epoch').mean()
# df = df.drop('epoch',axis=1)
df_plot = df_plot.fillna(0)

# df_plot['10c kl'] = df_plot['10c kl']/(10*(10+1)/2) 
# df_plot['20c kl'] = df_plot['20c kl']/(20*(20+1)/2)
# df_plot['50c kl'] = df_plot['50c kl']/(50*(50+1)/2)
# df_plot['70c kl'] = df_plot['70c kl']/(70*(70+1)/2)
# df_plot['VAE kl'] = df_plot['VAE kl']/(100)


# df_plot['10c rec'] = df_plot['10c rec']/(10*(10+1)/2) 
# df_plot['20c rec'] = df_plot['20c rec']/(20*(20+1)/2)
# df_plot['50c rec'] = df_plot['50c rec']/(50*(50+1)/2)
# df_plot['70c rec'] = df_plot['70c rec']/(70*(70+1)/2)
# df_plot['VAE rec'] = df_plot['VAE rec']/(100)

# df_plot['10c total loss'] = df_plot['10c total loss']/(10*(10+1)/2) 
# df_plot['20c total loss'] = df_plot['20c total loss']/(20*(20+1)/2)
# df_plot['50c total loss'] = df_plot['50c total loss']/(50*(50+1)/2)
# df_plot['70c total loss'] = df_plot['70c total loss']/(70*(70+1)/2)
# df_plot['VAE total loss'] = df_plot['VAE total loss']/(100)

fig = px.line(df_plot,x=df_plot.index,y=df.drop('epoch',axis=1).columns, title='Loss Metrics',color_discrete_sequence=['#636EFA', '#EF553B', '#00CC96', '#AB63FA','#FFA15A'])
fig.write_html("./150epochloss cifar10 rec.html")

pio.write_image(fig, './150epochloss rec.pdf', width=1600, height=1200)