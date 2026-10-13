"""CPU forward/backward and upstream segment behavior; no dataset downloads."""
import json, pickle, random, tempfile
from pathlib import Path
import torch
from dattri.benchmark.models.MusicTransformer.music_transformer import MusicTransformer
from dattri.benchmark.models.MusicTransformer.dataset.e_piano import EPianoDataset
from dattri.benchmark.models.MusicTransformer.utilities.constants import VOCAB_SIZE

torch.set_num_threads(2)
torch.manual_seed(42)
model=MusicTransformer(n_layers=1,num_heads=2,d_model=16,dim_feedforward=32,
                       max_sequence=16,rpr=True,dropout=0,device='cpu')
x=torch.randint(0,388,(2,16)); y=torch.randint(0,388,(2,16))
out=model(x)
assert out.shape==(2,16,VOCAB_SIZE)
loss=torch.nn.functional.cross_entropy(out.flatten(0,1),y.flatten())
loss.backward()
assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
torch.optim.Adam(model.parameters(),lr=1e-4).step()
with tempfile.TemporaryDirectory() as temp:
    with open(Path(temp)/'a.pickle','wb') as f: pickle.dump(list(range(600)),f)
    dataset=EPianoDataset(temp,max_seq=256,full_version=True)
    random.seed(1); a=dataset[0][0].clone()
    random.seed(2); b=dataset[0][0].clone()
    assert not torch.equal(a,b), 'Expected to reproduce offset-zero random-crop issue'
    assert dataset[1][0][0].item()==256
    assert len(dataset)==2
report={'torch':torch.__version__,'cuda_available':torch.cuda.is_available(),
        'vocabulary':VOCAB_SIZE,'output_shape':list(out.shape),'loss':loss.item(),
        'backward_and_optimizer':'passed','offset_zero_random_crop_reproduced':True}
Path('output').mkdir(parents=True,exist_ok=True)
Path('output/music_smoke.json').write_text(json.dumps(report,indent=2))
print(json.dumps(report,indent=2))
