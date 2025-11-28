import importlib.util
import json
import pathlib
import sys
import types


def _stub_module(name: str, attrs: dict | None = None):
    module = types.ModuleType(name)
    module.__path__ = []  # make it package-like for nested imports
    if attrs:
        for key, val in attrs.items():
            setattr(module, key, val)
    sys.modules[name] = module
    return module


habitat_sim_physics_stub = _stub_module(
    "habitat_sim.physics",
    {"ManagedArticulatedObject": type("ManagedArticulatedObject", (), {}), "ManagedRigidObject": type("ManagedRigidObject", (), {})},
)
habitat_sim_stub = _stub_module(
    "habitat_sim", {"geo": types.SimpleNamespace(Ray=lambda *a, **k: None), "stage_id": 0}
)
setattr(habitat_sim_stub, "Simulator", type("Simulator", (), {}))
setattr(habitat_sim_stub, "physics", habitat_sim_physics_stub)
setattr(
    habitat_sim_stub,
    "metadata",
    types.SimpleNamespace(MetadataMediator=type("MetadataMediator", (), {})),
)
_stub_module("habitat", {})
_stub_module("habitat.core", {})
_stub_module(
    "habitat.core.logging",
    {
        "logger": types.SimpleNamespace(
            info=lambda *a, **k: None, debug=lambda *a, **k: None, error=lambda *a, **k: None
        )
    },
)
_stub_module("habitat.datasets", {})
_stub_module("habitat.datasets.rearrange", {})
_stub_module("habitat.datasets.rearrange.samplers", {})
_stub_module("habitat.datasets.rearrange.samplers.receptacle", {"Receptacle": type("Receptacle", (), {})})
_stub_module(
    "habitat.datasets.rearrange.navmesh_utils",
    {
        "unoccluded_navmesh_snap": lambda *a, **k: None,
        "snap_point_is_occluded": lambda *a, **k: False,
    },
)
_stub_module("habitat.sims", {})
_stub_module("habitat.sims.habitat_simulator", {})
_stub_module(
    "habitat.sims.habitat_simulator.sim_utilities",
    {
        "get_obj_from_handle": lambda *a, **k: None,
        "obj_next_to": lambda *a, **k: None,
        "snap_down": lambda *a, **k: None,
        "get_ao_default_link": lambda *a, **k: None,
        "get_global_keypoints_from_object_id": lambda *a, **k: None,
        "get_obj_size_along": lambda *a, **k: None,
        "link_is_open": lambda *a, **k: False,
    },
)
_stub_module(
    "habitat.sims.habitat_simulator.object_state_machine",
    {"BooleanObjectState": object},
)
_stub_module("habitat.config", {})
_stub_module(
    "habitat.config.default_structured_configs",
    {
        "HabitatConfigPlugin": type("HabitatConfigPlugin", (), {}),
        "AgentConfig": type("AgentConfig", (), {}),
        "register_hydra_plugin": lambda *a, **k: None,
    },
)
_stub_module(
    "habitat_baselines.config.default_structured_configs",
    {"HabitatBaselinesConfigPlugin": type("HabitatBaselinesConfigPlugin", (), {})},
)
_stub_module("habitat_baselines.config", {})
_stub_module("habitat.tasks", {})
_stub_module("habitat.tasks.rearrange", {})
_stub_module("habitat.tasks.rearrange.rearrange_sim", {})
_stub_module("habitat.tasks.rearrange.rearrange_task", {})
_stub_module(
    "habitat.tasks.rearrange.rearrange_grasp_manager",
    {"RearrangeGraspManager": type("RearrangeGraspManager", (), {})},
)
_stub_module(
    "habitat.tasks.rearrange.articulated_agent_manager",
    {"ArticulatedAgentManager": type("ArticulatedAgentManager", (), {})},
)
_stub_module("habitat.articulated_agents", {})
magnum_stub = _stub_module("magnum", {})
setattr(magnum_stub, "Vector3", type("Vector3", (), {}))
setattr(magnum_stub, "Quaternion", type("Quaternion", (), {}))
setattr(magnum_stub, "Matrix4", type("Matrix4", (), {}))

_concept_path = pathlib.Path(__file__).resolve().parent.parent / "perception" / "concept_mapping.py"
_spec = importlib.util.spec_from_file_location("concept_mapping", _concept_path)
concept_mapping = importlib.util.module_from_spec(_spec)
assert _spec and _spec.loader
_spec.loader.exec_module(concept_mapping)
map_detection_to_concepts = concept_mapping.map_detection_to_concepts
entity_module = importlib.import_module("habitat_llm.world_model.entity")
world_graph_module = importlib.import_module("habitat_llm.world_model.world_graph")
Concept = entity_module.Concept
Object = entity_module.Object
WorldGraph = world_graph_module.WorldGraph


def test_add_or_update_concept_annotation():
    wg = WorldGraph()
    obj = Object("obj", {"type": "mug"})
    wg.add_node(obj)

    concept_node = wg.add_or_update_concept_annotation(obj, ["cup"], [0.8])

    assert isinstance(concept_node, Concept)
    assert obj.properties["concept_labels"] == ["cup"]
    assert obj.properties["concept_confidence"] == [0.8]
    assert concept_node in wg.graph
    assert wg.graph[concept_node][obj] == "describes"

    # updating with a stronger confidence should replace the stored value
    wg.add_or_update_concept_annotation(obj, ["cup"], [0.9])
    assert obj.properties["concept_confidence"] == [0.9]


def test_serialize_and_log_concept_layer(tmp_path):
    wg = WorldGraph()
    obj = Object("obj", {"type": "mug"})
    wg.add_node(obj)
    wg.add_or_update_concept_annotation(obj, ["cup"], [0.5])

    serialized = wg.serialize_concept_layer()
    assert serialized["entity_concepts"][0]["concept_confidence"] == [0.5]

    log_path = tmp_path / "concepts.json"
    wg.log_concept_layer(str(log_path))
    logged = json.loads(log_path.read_text())
    assert logged["entity_concepts"][0]["concept_labels"] == ["cup"]


def test_map_detection_to_concepts():
    labels, confidences = map_detection_to_concepts("chair", metadata=None)
    assert labels == ["chair"]
    assert confidences == [1.0]
