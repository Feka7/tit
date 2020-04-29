import pandas as pd
import matplotlib.pyplot as plt

#Semplici grafici per analizzare il problema. Dato che la maggior parte
#dei dati contiene valori discreti appartenenti ad insieme relativamente piccoli,
#non è stata data una particolare rilevanza a questa fase.
#In ogni caso, lo studio della distribuzione relativa alla tariffa di viaggio 
#può portare ad una migliore divisione in intervalli discreti di tale dato.

df = pd.read_csv("train.csv")

#costo biglietto
df.sort_values(by="Fare", ascending=True, inplace=True)
s = df['Fare']
t = []
for x in s:
    t.append(x)
plt.plot(t)
plt.ylabel('some numbers')
plt.title("costo biglietto")
plt.show()

plt.hist(t, 400)
plt.title("Distribuzione costo biglietto")
plt.show()


#sopravissuti

f_df = df[ df['Sex'] == "female"]
m_df = df[ df['Sex'] == "male"]

f_n = f_df['Sex']
tot_s = []
for x in f_n:
    tot_s.append(1)

m_n = m_df['Sex']
for x in m_n:
    tot_s.append(0)

f = f_df['Survived']
f_list = []
for x in f:
    f_list.append(x)

m = m_df['Survived']
m_list = []
for x in m:
    m_list.append(x)

plt.figure(figsize=(9, 5))
plt.subplot(131)
plt.gca().set_title('Rapporto uomo/donna')
plt.hist(tot_s, 3)
plt.subplot(132)
plt.gca().set_title('Sopravissute(=1) donne')
plt.hist(f_list, 3)
plt.subplot(133)
plt.gca().set_title('Sopravissuti(=1) uomini')
plt.hist(m_list, 3)
plt.suptitle('Sopravissuti in base al sesso')
plt.show()

c1_df = df[ df["Pclass"] == 1]
c1 = c1_df['Survived']
c1_list = []
for x in c1:
    c1_list.append(x)

c2_df = df[ df["Pclass"] == 2]
c2 = c2_df['Survived']
c2_list = []
for x in c2:
    c2_list.append(x)

c3_df = df[ df["Pclass"] == 3]
c3 = c3_df['Survived']
c3_list = []
for x in c3:
    c3_list.append(x)

plt.figure(figsize=(9, 5))
plt.subplot(131)
plt.gca().set_title('Prima classe')
plt.hist(c1_list, 3)
plt.subplot(132)
plt.gca().set_title('Seconda classe')
plt.hist(c2_list, 3)
plt.subplot(133)
plt.gca().set_title('Terza classe')
plt.hist(c3_list, 3)
plt.suptitle('Sopravissuti(=1) in base alla classe')
plt.show()
