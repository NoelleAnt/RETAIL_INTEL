from django import forms

class TransactionForm(forms.Form):
    age = forms.IntegerField(
        label="Age",
        min_value=18,
        max_value=100,
        widget=forms.NumberInput(attrs={"class": "form-control"})
    )
    quantity = forms.IntegerField(
        label="Quantity",
        min_value=1,
        widget=forms.NumberInput(attrs={"class": "form-control"})
    )
    price_per_unit = forms.FloatField(
        label="Price per Unit ($)",
        min_value=0,
        widget=forms.NumberInput(attrs={"class": "form-control", "step": "0.01"})
    )
    gender = forms.ChoiceField(
        label="Gender",
        choices=[("Male", "Male"), ("Female", "Female")],
        widget=forms.Select(attrs={"class": "form-control"})
    )
    product_category = forms.ChoiceField(
        label="Product Category",
        choices=[("Beauty", "Beauty"), ("Clothing", "Clothing"), ("Electronics", "Electronics")],
        widget=forms.Select(attrs={"class": "form-control"})
    )