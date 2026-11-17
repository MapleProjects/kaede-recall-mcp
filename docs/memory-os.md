# Memory OS

Capa de memoria tipada sobre una sola colección del backend vectorial. Reemplaza el índice
de skills en el prompt y la memoria plana en archivos por recuperación bajo
demanda, con presupuesto de bytes y relaciones dentro del payload.

## El problema que resuelve

- El índice de skills en el prompt cuesta tokens en cada turno y el modelo
  decide si cargar el cuerpo. La tasa de acierto es baja y el prompt crece.
- La memoria en archivos se inyecta completa o no se inyecta.
- Una búsqueda vectorial sin tope devuelve decenas de KB y ese bloque se
  reenvía en cada llamada posterior del mismo turno.
- Las relaciones entre decisiones, problemas y servicios no existen en un
  almacén puramente vectorial.

## Una colección, un vector

Colección `kaede-memory`.

- `dense`. Embedding del texto disparador, no del documento completo.
- Las relaciones viven en el payload como `edges` y `backlinks`. No hay puntos
  extra ni un vector de arista.

El texto disparador es la clave. Para un procedimiento es el nombre más la
descripción más la sección "When to use". Para un hecho es el resumen más los
primeros 1,200 caracteres. Embeber el cuerpo completo diluye la señal y fue la
causa medida de que el registro de cuentas no apareciera en el puesto uno.

## Payload

| Campo | Uso |
|---|---|
| `kind` | fact, preference, profile, decision, problem, procedure, trace, doc |
| `text` | Cuerpo completo, indexado full-text |
| `summary` | Texto disparador corto |
| `entities` | Alias declarados, en minúsculas, para consultas por entidad |
| `edges` | `[{rel, to}]` hacia otros puntos |
| `backlinks` | `[{rel, from}]` desde otros puntos |
| `status` | active o superseded |
| `deprecated` | Excluido por defecto, usado para snapshots viejos |
| `scope` | global o project:<id> |
| `hash` | Deduplicación por contenido |

Indexado: `kind`, `scope`, `status`, `project`, `entity`, `repo`, `host`,
`tags`, `rel`, `procedure_id`, `hash`, `source`, `superseded_by`, `entities`,
`deprecated`, las cuatro fechas y `text` con índice full-text.

## Entidades dinámicas

No hay diccionario de proyectos ni sinónimos en el código. Los alias viven en
el campo `entities` de cada punto.

- Al arrancar, `EntityRegistry` lee todos los valores únicos con la API `facet`
  del backend vectorial.
- En cada escritura se extraen identificadores estructurales del texto.
  kebab-case, snake_case, CamelCase, alfanumérico con dígitos y siglas.
- Los identificadores extraídos se fusionan con los declarados y se guardan en
  el payload. El registro crece en caliente con cada escritura.
- El servicio del hook recarga el registro cada 60 segundos.

Un proyecto nuevo entra al slotting por el solo hecho de escribirse. No hay
constantes que actualizar.

## Lectura

`recall` vectoriza la consulta una sola vez y reutiliza ese vector en dos
búsquedas.

1. Global. Sin filtro de entidad.
2. Slot de entidad. Solo si el registro encontró alias en la consulta. Mismo
   vector, filtro `entities` con esos alias.

Después aplica el bonus aditivo. +0.04 por el primer alias compartido, +0.02
por un segundo, tope +0.06, solo si el score base es mayor o igual a 0.30.

Armado de slots, sin análisis gramatical.

- Slot 1. Mejor nodo cuya entidad coincida con la consulta. Si no hay match, el
  mejor del ranking.
- Slot 2. Mejor nodo de un kind distinto, de la misma entidad si existe. Si no
  alcanza 0.30 de score base, no se reserva.
- Si el slot 2 supera al slot 1 por 0.01, se intercambian.
- El resto sigue el ranking global.

Cuerpo dual. Se abre el segundo cuerpo cuando el segundo supera 0.33 y se
cumple una de tres. Empate técnico con brecha menor o igual a 0.015. Par
complementario, procedimiento con hecho, problema, decisión o traza, con
brecha menor o igual a 0.046. O par de alta confianza, mismo tema, segundo
mayor o igual a 0.60 y brecha menor o igual a 0.09.

Salida progresiva. Dos cuerpos de 450 caracteres cuando hay dual, uno de 60 por
ciento del presupuesto cuando no. Del tercero al quinto, solo `id`, `trigger` y
`score`. Los nodos con score menor a 0.20 se descartan.

Filtro duro antes de buscar. `status=superseded` y `deprecated=true` quedan
fuera con `must_not`.

Modo `ids`. `memory-recall` acepta una lista de ids y hace `retrieve` sin
vectorizar.

## Escritura

- Hash de contenido para deduplicar.
- Contradicción no borra. Marca `status=superseded`, `valid_to` y agrega la
  arista `superseded_by` en ambos payloads.
- `links` en la misma llamada. Formato `rel:id,rel:id`. Escribe ida y vuelta.
- `entities` en la misma llamada. Formato `rpi,raspberry-pi,systemd`.

## Herramientas MCP

Dos, no cinco. Se activan con `MEMORY_OS_ENABLED=true`. Las seis originales
siguen disponibles para acceso crudo.

- `memory-recall(query, ids, kinds, scope, project, limit, max_chars, walk_depth, include_superseded)`.
  Búsqueda con presupuesto, slot de entidad y salto de grafo opcional. Con
  `ids` abre cuerpos sin vectorizar.
- `memory-write(kind, text, summary, scope, project, tags, entities, links, supersedes_id, source)`.
  Escribe, enlaza y reemplaza en una sola llamada.

## CLI

```bash
python -m kaede_recall_mcp.memory_os.bootstrap create
python -m kaede_recall_mcp.memory_os.bootstrap migrate --source agents-memory-v2
python -m kaede_recall_mcp.memory_os.bootstrap skills
python -m kaede_recall_mcp.memory_os.bootstrap query "crear cuenta en un portal"
python -m kaede_recall_mcp.memory_os.bootstrap stats
```

## Medición

Modelos sobre 228 skills con texto disparador, posición del procedimiento
correcto.

| Modelo | cuenta | saturación | rpi | flutter |
|---|---|---|---|---|
| jina-v2-base-es | 1 | 1 | 1 | 1 |
| multilingual-e5-large | 1 | 1 | 6 | 1 |
| jina-embeddings-v3 | 1 | 1 | 1 | 1 |

Colección completa, 1,793 puntos, antes y después del slotting con entidades.

| Consulta | Antes | Después |
|---|---|---|
| crear cuenta en un sitio web | procedimiento 1 | procedimiento 1, cuerpo |
| saturación de pantalla en Hyprland | hecho 1, procedimiento 2 | procedimiento 1, cuerpo |
| el servicio del RPi no responde | Flatpak 1, NAT en 9 | NAT 1, Flatpak 2 |
| por qué falló el complexivo | teclado Hyprland 1, complexivo 2 | complexivo 1, cuerpo completo |
| publicar app Flutter en Android | procedimiento 1 | procedimiento 1, cuerpo |

### Calibración de 30 pares

Cinco temas por seis paráfrasis, con el id esperado anotado a mano.

- Puesto 1 en 24 de 30.
- Puesto 2 o mejor en 29 de 30.
- Percentil 25 de la brecha entre el primero y el segundo, 0.0472. El margen
  dual quedó en 0.047.
- Un solo caso fuera del top 2. El hecho del RPi queda detrás de otro hecho del
  mismo equipo y del procedimiento de auditoría. No hay tema equivocado.
- La consulta del RPi deja Flatpak en el slot 2 con score mayor. Es intencional.
  Un nodo global sin la entidad no destrona al nodo de la entidad, y la brecha
  de esa fila es la única negativa del CSV.
- Cero listas de verbos, sinónimos o proyectos en el código. El registro sale
  del backend vectorial y crece con cada escritura.

Experimentos medidos y descartados, no quedan en el código.

- BM25 sobre la consulta completa. Ordena por palabras compartidas y entierra
  el procedimiento correcto bajo correos y GitHub.
- Carril de términos raros por frecuencia de documento. Sube documentos que
  comparten una palabra pero no el tema.
- Boost multiplicativo por entidad. Un alias común declarado en pocos puntos
  desplaza al procedimiento correcto aunque el documento no sea del tema.
- Rerank con jina-v3 sobre los 12 primeros. Mejora una consulta y empeora dos.
  Sin ganancia neta, con un GB más de RAM y 12 segundos de latencia.

Coste medido por consulta. 3 resultados, entre 400 y 900 caracteres. Antes, una
búsqueda amplia devolvía 84,131 caracteres.

## Límites conocidos

- El slot 2 toma el mejor procedimiento global aunque la consulta no sea
  imperativa. En una pregunta de causa puede colar un procedimiento de otro
  tema. Medido y pendiente de restringir a procedimientos de la entidad.
- El denso confunde un hecho específico con un procedimiento general cuando
  ambos son relevantes. La solución pendiente es un planificador de consulta
  que separe entidad e intención antes de buscar, no otro modelo de embedding.
- El recorrido de grafo es por niveles, un retrieve por salto. Sirve para
  grafos personales de miles de aristas.
- No hay transacción entre escrituras. Una operación compuesta se reintenta por
  id determinista.
- La consolidación semanal de hechos todavía es manual.
- El enrutado automático antes del turno no está en esta capa.
