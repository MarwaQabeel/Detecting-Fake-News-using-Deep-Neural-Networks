from django.contrib import admin ; # type: ignore
from django.urls import re_path ; # type: ignore
from repository.views import *


app_name = 'repository'

urlpatterns = [
    # re_path(r"^$", main_view, name="main"),
    re_path(r'^$', SearchView.as_view(), name='main'),
    re_path(r'^about/', about_view, name='about'),
    # re_path(r'^contact/', contact_view, name='contact'),
    re_path(r'^search/', search_view, name='search'),
    re_path(r'^autoComplete/', autoComplete, name='autoComplete'), 
]