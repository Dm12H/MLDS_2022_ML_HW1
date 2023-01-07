import re
import numpy as np
import itertools


def get_all_units(column):
    pattern = re.compile("[A-Za-z]+")

    def _get_units(x):
        if isinstance(x, float):
            return {}
        units_extracted = pattern.findall(str(x))
        units_lower = map(lambda x: x.lower(), units_extracted)
        return units_lower
    unitlist = column.apply(_get_units)
    units_flat = itertools.chain.from_iterable(unitlist)
    return set(units_flat)


def standardize_torque(val):
    if isinstance(val, float):
        return val
    merged_string = "".join(re.split(r'\s+|,', val)).lower()
    tilda_removed = merged_string.replace("~","-")
    standardized_string = re.sub(r"(at|/(?!-))", "@", tilda_removed)
    return standardized_string


def process_torque(df):
    kg_to_n = 9.80665
    re1 = re.compile("^(?P<torque>\d+(?:\.\d+)?)(?P<unit>nm|kgm|)(?:@(?P<rpm>\d+(?:(?:-|\+\/-)\d+)?)(?:rpm)?)?$")
    re2 = re.compile("^(?P<torque>\d+(?:\.\d+)?)@(?P<rpm>\d+(?:\+?\/?-?\d+)?)\((?P<unit>nm|kgm|)")
    re3 = re.compile("^(?P<torque>\d+)(?P<unit>nm|kgm|).*@(?P<rpm>\d+)")
    re_list = (re1, re2, re3)
    standardized_column = df.torque.map(standardize_torque)
    torque = []
    max_rpm = []
    for val in standardized_column:
        if val is np.nan:
            torque.append(val)
            max_rpm.append(val)
            continue
        for re_expr in re_list:
            match = re_expr.match(val)
            if match:
                break
        else:
            raise ValueError(f"following string not parsed '{val}''")
        torque_val = float(match.group("torque"))
        units = match.group("unit")
        # записи без единиц измерения считаем в ньютонах
        if units == "kgm":
            torque_val *= kg_to_n
        rpm_line = match.group("rpm")
        if rpm_line:
            try:
                rpm = float(rpm_line)
            except ValueError:
                lval, sep, rval = re.split(r"(\D+)", rpm_line)
                # берем более низкое значение rpm, т.к. кажется,
                # то же значение torque на низких оборотах лучше, чем на высоких
                if sep == "-":
                    rpm = float(lval)
                elif sep == "+/-":
                    rpm = float(lval) - float(rval)
                else:
                    raise ValueError(f"cannot parse rpm line: {rpm_line}")
        else:
            rpm = np.nan
        torque.append(torque_val)
        max_rpm.append(rpm)
    return torque, max_rpm


def clean_feature(entry):
    if entry is np.nan:
        return entry
    try:
        val, *_ = entry.strip().split(" ")
    except AttributeError:
        raise ValueError(f"val is not string:{entry}")
    if not re.match(r"\d+(?:.d+)?", val):
        return np.nan
    return float(val)


def clean_up_data(dataframe):
    for fname in ("mileage", "engine", "max_power"):
        dataframe[fname] = dataframe[fname].map(clean_feature)
    torque, max_rpm = process_torque(dataframe)
    dataframe["torque"] = torque
    dataframe["max_rpm"] = max_rpm
    return dataframe


def add_nonlinearity(df):
    df["km_driven_sq"] = df.km_driven.map(lambda x: x ** 2)
    df["year_squared"] = df.year.map(lambda x: x ** 2)
    return df


def get_model_info(column):
    maker, model, *_ = column.split(" ")
    return maker, model


class CarEncoder:
    def __init__(self, base_df):
        self.stats = {
            "all": base_df.selling_price.mean()
        }
        manufacturers, models = zip(*base_df.name.map(get_model_info))
        temp_df = base_df.assign(manufacturer=manufacturers, model=models)
        for manufacturer in temp_df.manufacturer.unique():
            manufacturer_slice = temp_df[temp_df.manufacturer == manufacturer]
            maker_price = manufacturer_slice.selling_price.mean()
            self.stats[manufacturer] = {"all": maker_price}
            for model in manufacturer_slice.model.unique():
                model_price = manufacturer_slice[manufacturer_slice.model == model].selling_price.mean()
                self.stats[manufacturer][model] = model_price

    def transform(self, df):
        manufacturer_correction = []
        model_correction = []

        avg = self.stats["all"]
        for manufacturer, model in df.name.map(get_model_info):
            manufacturer_stats = self.stats.get(manufacturer, None)
            if manufacturer_stats is None:
                manufacturer_correction.append(0)
                model_correction.append(0)
                continue
            manufac_avg = manufacturer_stats["all"]
            manufacturer_correction.append(manufac_avg - avg)
            model_avg = manufacturer_stats.get(model, None)
            if model_avg is None:
                model_correction.append(0)
            else:
                model_correction.append(model_avg - manufac_avg)
        processed_df = df.assign(manufacturer=manufacturer_correction, model=model_correction)
        return processed_df.drop(columns="name")


class PredictionModel:

    def __init__(self, model, encoder, medians):
        self.model = model
        self.car_encoder = encoder
        self.medians = medians

    def predict(self, df):
        clean_data = clean_up_data(df)
        nonlinear_data = add_nonlinearity(clean_data)
        finalized_data = self.car_encoder.transform(nonlinear_data)
        finalized_data.fillna(value=self.medians, inplace=True)
        prediction = self.model.predict(finalized_data)
        return prediction
