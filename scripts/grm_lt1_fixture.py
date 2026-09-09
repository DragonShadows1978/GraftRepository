"""LT1 frozen scripted dialogue. Prior art: GRM C7 fixture scheduling (2026),
borrow disjoint source/probe placement; original natural-language content.
No prior art known to me for this exact dialogue. Not a memory algorithm.
"""
from pathlib import Path
import json
from scripts.grm_c7_common import create, sha
ROOT = Path(__file__).resolve().parents[1]

def build():
    events, probes, chains = {}, [], {}
    def fact(t, entity, attr, value, kind='fact', prefix=''):
        key = entity + '/' + attr
        old = chains.get(key, [])
        text = f"{prefix}{entity}'s {attr} will be {value}."
        if kind == 'correction':
            text = f"Actually, {entity}'s {attr} will be {value}, replacing the earlier choice."
        events[t] = dict(turn=t, kind=kind, user=text,
            assistant={'fact':"That gives us something concrete to design around.",
                       'correction':"Understood; I'll use the revised choice."}[kind],
            entity=entity, attribute=attr, value=value, previous_turn=old[-1] if old else None)
        chains.setdefault(key, []).append(t)
        return key
    k0=fact(1,'Vega','docking fee','17 credits')
    k1=fact(2,'Medibay','bed count','12 beds')
    fact(3,'Vega','docking fee','23 credits','correction')
    fact(4,'Medibay','bed count','14 beds','correction')
    k2=fact(5,'Kestrel','cargo allowance','37 crates')
    k3=fact(6,'Lantern','launch date','18 October 2196')
    specs=[('fresh','Promenade','lamp spacing','9 voxels'),
           ('fresh','Commtower','maintenance crew','Iona Vale'),
           ('fresh','Breakwater','map position','(-31, 48, 12)'),
           ('correction','Vega','docking fee','29 credits'),
           ('correction','Medibay','bed count','16 beds'),
           ('alias','Kestrel','cargo allowance','37 crates'),
           ('alias','Lantern','launch date','18 October 2196')]
    occupied=set(events)|{200}
    for j,(cls,entity,attr,value) in enumerate(specs):
        anchor=next(t for t in range(12,50) if not ({t,*[t+d for d in (10,25,50,100,150)]}&occupied))
        occupied.update({anchor,*[anchor+d for d in (10,25,50,100,150)]})
        key=entity+'/'+attr
        if cls=='alias':
            alias=['the Hauler','the Beacon'][j-5]
            events[anchor]=dict(turn=anchor,kind='alias',entity=entity,alias=alias,
                user=f"Let's call {entity} '{alias}' from now on; it sounds more like a place our crew would use.",
                assistant="I like that shorthand. I'll use it in our discussion.")
            source=chains[key]+[anchor]
            target=alias
        else:
            fact(anchor,entity,attr,value,cls if cls=='correction' else 'fact')
            source=list(chains[key]); target=entity
        for d in (10,25,50,100,150):
            t=anchor+d
            question=f"What did we settle on for {target}'s {attr}?"
            p=dict(id=f'recall_{j+1}_{d}',turn=t,**{'class':cls},distance=d,
                source_turns=source,source_turn=anchor,question=question,expected=value,
                entity=entity,attribute=attr,answerable=True)
            probes.append(p)
            events[t]=dict(turn=t,kind='probe',user=question,probe_id=p['id'])
    # Side decisions are intentionally meaningful but not all are queried.
    side=[
      ('Foundry','iron reserve','83 ingots'),('Orchard','water reserve','146 litres'),
      ('Aster','survey lead','Neri Sol'),('Tern','repair budget','412 credits'),
      ('Morrow','arrival date','22 November 2196'),('Spindle','map position','(62, -17, 8)'),
      ('Quarry','drill count','7 drills'),('Saffron','galley stock','52 meals'),
      ('Harbor','rescue pilot','Edda Rook'),('Relay','battery reserve','31 cells'),
      ('Ferry','seat count','11 seats'),('Garden','seed stock','93 packets'),
      ('Kiln','heat limit','680 kelvin'),('Shelter','oxygen reserve','245 litres'),
      ('Cairn','beacon height','27 voxels'),('Sparrow','rental price','61 credits'),
      ('Rill','pump count','4 pumps'),('Marrow','supply date','9 December 2196'),
      ('Osprey','navigator','Tavi Moss'),('Cobalt','map position','(-72, 15, -6)'),
      ('Gull','sensor count','13 sensors'),('Cinder','fuel stock','118 canisters'),
      ('Thistle','botanist','Lysa Fen'),('Flint','tool price','34 credits'),
      ('Hearth','opening date','6 January 2197'),('Cove','map position','(19, 83, -22)'),
      ('Wren','drone count','8 drones'),('Petrel','wire stock','267 metres'),
      ('Ember','quartermaster','Oren Pike'),('Willow','permit price','46 credits'),
      ('Mesa','inspection date','13 February 2197'),('Basin','map position','(-9, -44, 71)'),
      ('Lark','locker count','24 lockers'),('Reef','glass stock','156 panels'),
      ('Dune','medic','Sela Voss'),('Crag','tug price','385 credits'),
      ('Glade','festival date','27 March 2197'),('Vale','map position','(55, -26, 39)'),
      ('Plover','airlock count','3 airlocks'),('Drift','cable stock','179 metres'),
      ('Fallow','steward','Arlo Reed'),('Nacre','ticket price','19 credits'),
      ('Bracken','handover date','11 April 2197'),('Shoal','map position','(-63, 28, 46)'),
      ('Finch','suit count','18 suits'),('Grove','soil stock','224 sacks'),
      ('Russet','architect','Mira Holt'),('Brook','freight price','72 credits'),
      ('Slate','test date','16 May 2197'),('Rookery','map position','(41, 66, -13)'),
      ('Nettle','filter count','32 filters'),('Weir','spare stock','57 valves'),
      ('Mossbank','curator','Kavi Elm')]
    free=[t for t in range(30,180) if t not in events]
    # Evenly space additional decisions; no run-time routing intervention.
    slots=[free[round(i*(len(free)-1)/(len(side)-1))] for i in range(len(side))]
    for t,spec in zip(slots,side): fact(t,*spec)
    extra=[('Foundry','iron reserve','89 ingots'),('Foundry','iron reserve','91 ingots'),
           ('Orchard','water reserve','152 litres'),('Aster','survey lead','Neri Ash'),
           ('Tern','repair budget','438 credits'),('Morrow','arrival date','24 November 2196'),
           ('Spindle','map position','(64, -17, 8)'),('Quarry','drill count','9 drills'),
           ('Saffron','galley stock','58 meals'),('Harbor','rescue pilot','Edda Fern'),
           ('Relay','battery reserve','35 cells')]
    for spec in extra:
        prev=chains[spec[0]+'/'+spec[1]][-1]
        t=next(t for t in range(prev+3,190) if t not in events)
        fact(t,*spec,kind='correction')
    for entity,alias in [('Foundry','the Forge'),('Orchard','the Cistern'),('Aster','the Lookout'),
                         ('Tern','the Workshop'),('Morrow','the Gate'),('Spindle','the Needle'),
                         ('Quarry','the Pit'),('Saffron','the Kitchen')]:
        prev=max(e['turn'] for e in events.values() if e.get('entity')==entity)
        t=next(t for t in range(prev+2,191) if t not in events)
        events[t]=dict(turn=t,kind='alias',entity=entity,alias=alias,
            user=f"Let's call {entity} '{alias}' from now on. That fits its role in the expansion.",
            assistant="That should feel comfortable in crew conversation.")
    ordinary=[
      ('I want the arrival to feel welcoming rather than overwhelming.','Let the promenade reveal itself as the player walks out.'),
      ('Can we make the voxel construction feel handmade?','Small irregularities in the silhouettes could suggest people built it.'),
      ('The Galactic Hub should still feel like a social place.','We can give people reasons to linger without asking them to fight.'),
      ('I keep picturing a tired pilot stopping for tea.','That is a useful mood for the quieter corners.'),
      ('How do we avoid turning the medibay into a gloomy corridor?','Give it warm materials and a view toward a public space.'),
      ('The comm tower ought to feel useful without dominating the skyline.','Its silhouette can be recognizable while leaving room for the rest of the station.'),
      ('I like seeing other ships through the docking windows.','It makes the settlement feel connected to a larger world.'),
      ('Could the salvage loop encourage curiosity?','Let unusual shapes suggest possibilities before showing a crafting menu.'),
      ('I worry that too much signage will hide the architecture.','Use sightlines and material changes before adding signs.'),
      ('Sometimes I enjoy just wandering around a game.','We should leave spaces where wandering is rewarding on its own.'),
      ('Would a silent observation room be too uneventful?','A calm view can provide contrast with the machinery outside.'),
      ('The cargo areas should look busy without becoming confusing.','Clear paths and readable stacks would help.'),
      ('I would rather discover a shortcut than follow a glowing trail.','A glimpse through an opening can invite exploration.'),
      ('Let us keep the voxel scale visible in the walls.','Then repairs and additions can remain part of the visual story.'),
      ('I am taking a quick tea break before we continue.','Good moment to step back and imagine walking through the place.'),
      ('How should the sound change near an airlock?','A shift in ambience can signal the boundary before the door moves.'),
      ('I like the idea that crews leave traces of everyday life.','Personal clutter can imply routines without exposition.'),
      ('Would weathered paint work on the freight decks?','It could show repeated handling and make the materials readable.'),
      ('This expansion needs room for players to make their own stories.','We can provide places and tools without dictating every outcome.'),
      ('I do not want every room to contain a reward chest.','Some rooms can simply make the station believable.'),
      ('Could we make repair work satisfying to watch?','Visible changes to damaged surfaces would connect the action to its result.'),
      ('The promenade needs places where people can stop without blocking traffic.','Recessed seating could preserve a clear walking route.'),
      ('How do we keep the no-combat space interesting?','Movement, conversation, and small environmental changes can carry it.'),
      ('I keep coming back to the view of arriving ships.','It gives the player a sense of motion even while standing still.'),
      ('We should not make the tutorial explain every machine.','Let familiar shapes and gentle experimentation do some of that work.'),
      ('Is it all right if the outer walk feels a little lonely?','That could make the return to the social areas feel warmer.'),
      ('The menus should not interrupt the feeling of being there.','Brief interactions can leave more attention on the surroundings.'),
      ('I would like cargo to have a convincing sense of weight.','Animation and sound can help even with blocky geometry.'),
      ('What makes a maintenance corridor worth exploring?','Hints of how the station works can be more interesting than another prize.'),
      ('I think the distant engine hum should be almost comforting.','A steady ambience can tie otherwise different spaces together.'),
      ('The medibay and docks need to feel part of the same settlement.','Shared materials can establish that connection without identical layouts.'),
      ('I am trying to imagine this in a long evening session.','Alternating busy and quiet spaces should give the player room to breathe.'),
      ('Could the player recognize a room just from its doorway?','A distinct silhouette or lighting rhythm could make that possible.'),
      ('I do not want a beautiful station that is annoying to navigate.','We should judge it by walking routes as well as screenshots.'),
      ('Let us think about the experience of returning after an expedition.','The transition from exposed space to familiar shelter could matter a lot.'),
      ('I like scuffed surfaces more than perfectly polished ones here.','They suggest continued use and give the lighting something to catch.'),
      ('The tower could be a meeting point even for people who never use its terminal.','Its surrounding space can support that social role.'),
      ('It would be nice to hear activity without seeing its source immediately.','That can make neighboring rooms feel connected.'),
      ('I think we have the mood now.','Yes, the practical choices can support that mood rather than compete with it.'),
      ('Let us keep the final walkthrough focused on how the place feels.','We can pause wherever the layout loses its sense of welcome.')]
    empties=[t for t in range(1,200) if t not in events]
    # Second pass revisits themes with different natural responses; no facts.
    for i,t in enumerate(empties):
        u,a=ordinary[i%len(ordinary)]
        if i>=len(ordinary):
            u='Coming back to that earlier thought: '+u[0].lower()+u[1:]
            a='That still fits the direction we have been discussing. '+a
        events[t]=dict(turn=t,kind='ordinary',user=u,assistant=a)
    for p in probes: p['oracle_source_texts']=[events[t]['user'] for t in p['source_turns']]
    decisions=[dict(entity=p['entity'],attribute=p['attribute'],expected=p['expected'],source_turns=p['source_turns'])
               for p in probes if p['distance']==150][:5]
    events[200]=dict(turn=200,kind='recap',user='recap the five biggest decisions we made')
    return dict(schema='grm.lt1.fixture.v1',turns=[events[t] for t in range(1,201)],
        probes=sorted(probes,key=lambda p:p['turn']),decisions=decisions,restart_after_turns=[70,140],
        recency_mounts=2,distances=[10,25,50,100,150],
        protocol='Frozen user/assistant dialogue replay; recall and recap responses generated. Probe answers not deposited. Natural corrections only, no hidden memory commands.',
        provenance='Invented expansion facts; vocabulary from Project-Frontier docs/PLAN_E_HUB.md. Not canonical game changes.')

def main():
    f=build(); out=ROOT/'fixtures/lt1'
    create(out/'dialogue.json',f)
    with (out/'turn_plan.md').open('x') as stream:
        stream.write('# LT1 frozen conversation\n\n'+f['protocol']+'\n\n')
        for e in f['turns']:
            stream.write(f"{e['turn']}. **{e['kind']} — User:** {e['user']}\n\n")
            stream.write('   **Assistant:** '+e.get('assistant','[generated at execution]')+'\n\n')
    from collections import Counter
    create(out/'manifest.json',dict(files={n:sha(out/n) for n in ['dialogue.json','turn_plan.md']},
        turn_mix=dict(Counter(e['kind'] for e in f['turns'])),turns=200,recalls=35,recap=1))
    (out/'manifest.sha256').write_text(sha(out/'manifest.json')+'  manifest.json\n')
if __name__=='__main__': main()
