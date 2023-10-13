{%- extends 'full.tpl' -%}

{%- block input_group -%}
  {%- if cell.metadata.hide_input -%}
  {%- else -%}
    {{ super() }}
  {%- endif -%}
{%- endblock input_group -%}
