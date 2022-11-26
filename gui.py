import enum
from tkinter import *
from tkinter import Tk

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

general_example_x = np.asarray([[-0.24216784, -0.25672372, -0.03388052, -0.30032498, -0.07865291, -0.52939804,
                                 0.24234088, -0.19052635, -0.26371173, -0.61144378, -1.05309294, -0.49518356]])
general_example_y = 2

data: pd.DataFrame = pd.read_excel('deers.xlsx', sheet_name='DATA')
df = data.drop(
    ['WIEK', 'P_CON_CO', 'CIEZARPO', 'P_M3Ż', 'FS_FS', 'DA_DA', 'N_RH', 'M1_M3', 'P_M3', 'P_ZL', 'P_B', 'P_CON_CO'],
    axis=1)
df['MIEJSCE'] = df['MIEJSCE'].map({'CZ': 0, 'B': 1, 'K': 2})
x = df.drop(["MIEJSCE"], axis=1)
y = df["MIEJSCE"]

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3)

scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)
x1 = df[df['KLASA'] == 1].drop(['MIEJSCE'], axis=1)
x2 = df[df['KLASA'] == 2].drop(['MIEJSCE'], axis=1)
x3 = df[df['KLASA'] == 3].drop(['MIEJSCE'], axis=1)
y1 = df[df['KLASA'] == 1]['MIEJSCE']
y2 = df[df['KLASA'] == 2]['MIEJSCE']
y3 = df[df['KLASA'] == 3]['MIEJSCE']

scaler = StandardScaler()
x1_train = scaler.fit_transform(x1)
x2_train = scaler.fit_transform(x2)
x3_train = scaler.fit_transform(x3)

df = df.fillna({
    "TUSZA": df.groupby('KLASA')["TUSZA"].transform("mean"),
    "ST_P": df.groupby('KLASA')["ST_P"].transform("mean"),
    "PM_P": df.groupby('KLASA')["PM_P"].transform("mean"),
    "ZY_ZY": df.groupby('KLASA')["ZY_ZY"].transform("mean"),
    "EU_EU": df.groupby('KLASA')["EU_EU"].transform("mean"),
    "CON_CON": df.groupby('KLASA')["CON_CON"].transform("mean"),
    "ID_GOC": df.groupby('KLASA')["ID_GOC"].transform("mean"),
    "P_OP": df.groupby('KLASA')["P_OP"].transform("mean"),
    "GOV_CR": df.groupby('KLASA')["GOV_CR"].transform("mean"),
    "C_P1": df.groupby('KLASA')["C_P1"].transform("mean")
})


class Affinity(enum.Enum):
    GENERAL = 'General'
    CZECH = 'Czech'


class TrainedModel(enum.Enum):
    RANDOM_FOREST = 'RandomForestClassifier'
    GAUSSIAN = 'GaussianNB'
    K_NEAREST = 'KNeighborsClassifier'
    SVC = 'SVC'
    BERNOULLI = 'BernoulliNB'
    DECISION_TREE = 'DecisionTreeClassifier'


def load_model(affinity: Affinity, modelName: TrainedModel = TrainedModel.RANDOM_FOREST):
    return joblib.load(str.join('/', ['models', affinity, modelName]))


class UserInterface:
    affinity_options = [
        "1",
        "2",
        "3"
    ]
    model_options = [
        "RandomForestClassifier",
        "GaussianNB",
        "KNeighborsClassifier",
        "SVC",
        "BernoulliNB",
        "DecisionTreeClassifier"
    ]
    regions_str = ["Czechy", "Beskidy", "Karpaty"]

    def classify(self):
        affinity = "Triple_Class_" + self.affinity_dropdown_value.get()
        model = load_model(affinity, self.model_dropdown_value.get())
        input_x = np.asarray([[float(self.affinity_dropdown_value.get()), float(self.tusza.get()),
                               float(self.p_op.get()), float(self.st_p.get()),
                               float(self.pm_p.get()), float(self.zy_zy.get()), float(self.ect_ect.get()),
                               float(self.eu_eu.get()), float(self.con_con.get()), float(self.id_goc.get()),
                               float(self.gov_cr.get()),
                               float(self.c_p1.get()), float(self.rok.get())]])
        xs = scaler.transform(input_x)
        prediction = model.predict(xs)

        self.display_var.set('Region: ' + self.regions_str[int(prediction[0])])

    def __init__(self):
        self.master = Tk()
        self.master.title("Jelenie")
        self.affinity_dropdown_value = StringVar()
        self.model_dropdown_value = StringVar()

        self.affinity_dropdown_label = Label(self.master, text="Klasa")
        self.affinity_dropdown_label.grid(row=0, column=5)
        self.affinity_dropdown_value.set("1")
        self.affinity_dropdown = OptionMenu(self.master, self.affinity_dropdown_value, *self.affinity_options)
        self.affinity_dropdown.grid(row=1, column=5)

        self.model_dropdown_label = Label(self.master, text="Model")
        self.model_dropdown_label.grid(row=2, column=5)
        self.model_dropdown_value.set("RandomForestClassifier")
        self.model_dropdown = OptionMenu(self.master, self.model_dropdown_value, *self.model_options)
        self.model_dropdown.grid(row=3, column=5)

        self.tusza_label = Label(self.master, text="Tusza").grid(row=4, column=0)
        self.tusza = Entry(self.master)
        self.tusza.grid(row=4, column=1)
        self.p_op_label = Label(self.master, text="P_OP").grid(row=4, column=2)
        self.p_op = Entry(self.master)
        self.p_op.grid(row=4, column=3)
        self.st_p_label = Label(self.master, text="ST_P").grid(row=4, column=4)
        self.st_p = Entry(self.master)
        self.st_p.grid(row=4, column=5)

        self.pm_p_label = Label(self.master, text="PM_P").grid(row=4, column=6)
        self.pm_p = Entry(self.master)
        self.pm_p.grid(row=4, column=7)
        self.zy_zy_label = Label(self.master, text="ZY_ZY").grid(row=4, column=8)
        self.zy_zy = Entry(self.master)
        self.zy_zy.grid(row=4, column=9)

        self.ect_ect_label = Label(self.master, text="ECT_ECT").grid(row=5, column=0)
        self.ect_ect = Entry(self.master)
        self.ect_ect.grid(row=5, column=1)

        self.eu_eu_label = Label(self.master, text="EU_EU").grid(row=5, column=2)
        self.eu_eu = Entry(self.master)
        self.eu_eu.grid(row=5, column=3)

        self.con_con_label = Label(self.master, text="CON_CON").grid(row=5, column=4)
        self.con_con = Entry(self.master)
        self.con_con.grid(row=5, column=5)
        self.id_goc_label = Label(self.master, text="ID_GOC").grid(row=5, column=6)
        self.id_goc = Entry(self.master)
        self.id_goc.grid(row=5, column=7)

        self.gov_cr_label = Label(self.master, text="GOV_CR").grid(row=5, column=8)
        self.gov_cr = Entry(self.master)
        self.gov_cr.grid(row=5, column=9)

        self.c_p1_label = Label(self.master, text="C_P1").grid(row=6, column=2)
        self.c_p1 = Entry(self.master)
        self.c_p1.grid(row=6, column=3)

        self.rok_label = Label(self.master, text="Rok").grid(row=6, column=4)
        self.rok = Entry(self.master)
        self.rok.grid(row=6, column=5)

        self.classify_button = Button(self.master, width=32, text="Classify", command=self.classify).grid(row=7,
                                                                                                          column=5)
        self.display_var = StringVar()
        self.display_lab = Label(self.master, textvariable=self.display_var).grid(row=11, column=5)
        self.master.mainloop()
