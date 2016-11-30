from django.conf.urls import include, url

from django.contrib import admin
admin.autodiscover()
from interface import views
from django.views.generic import TemplateView

urlpatterns = [
    url(r'^admin/', include(admin.site.urls)),
    url(r'^$', views.index, name='index'),
    url(r'^stats/', views.stats, name='stats'),
    url(r'^info/', views.info, name='info'),
    url(r'^docs/', views.docs, name='docs'),
    url(r'^example/', views.example, name='example'),
    url(r'^fit-predict/', views.fit_predict, name='fit-predict'),
    url(r'^robots\.txt$', TemplateView.as_view(template_name='robots.txt', content_type='text/plain')),
    url(r'^humans\.txt$', TemplateView.as_view(template_name='humans.txt', content_type='text/plain')),

]
