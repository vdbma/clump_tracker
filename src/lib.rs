use pyo3::prelude::*;

/// A Python module implemented in Rust. The name of this module must match
/// the `lib.name` setting in the `Cargo.toml`, else Python will not be able to
/// import the module.
#[pymodule]
mod _core {
    use pyo3::prelude::*;
    use pyo3::types::{PyDict,PyInt};
    use numpy::ndarray::{Array,ArrayRef, Array1,ArrayD, ArrayView1, ArrayViewD, ArrayViewMutD,meshgrid, MeshIndex, Axis, Zip,Slice};
    use numpy::{IntoPyArray, PyArray1, PyArrayDyn, PyReadonlyArrayDyn, PyArrayMethods};
    use numpy::convert::{ToPyArray};
    use std::any::type_name;
    fn type_of<T>(_: T) -> &'static str {
        type_name::<T>()
    }

    fn compute_total_energy_adi_shearing_box(rho: &ArrayViewD<'_, f64>,
                         vx1: &ArrayViewD<'_, f64>,
                         vx2: &ArrayViewD<'_, f64>,
                         vx3: &ArrayViewD<'_, f64>,
                         prs: &ArrayViewD<'_, f64>,
                         phi: &ArrayViewD<'_, f64>,
                         q: f64,
                         omega :f64,
                         gamma: f64,
                         x:&ArrayView1<'_, f64>,
                         y:&ArrayView1<'_, f64>,
                         z:&ArrayView1<'_, f64>) -> ArrayD< f64> {
                        
    let (xx,_yy,_zz) = meshgrid((x,y,z),MeshIndex::IJ); //requieres unreleased nupy-rust to acces ndarray >=0.17
    
    let e_c = 0.5 * rho * (vx1 * vx1 + (vx2 + q * omega * &xx) * (vx2 + q * omega * &xx) + vx3 * vx3);
    let e_th = prs / (gamma -1.);                        
    let e_p = rho * phi;
    &e_th + &e_c + &e_p
    }
    
    fn gradient(field: &ArrayViewD<'_, f64>, x: &ArrayView1<'_, f64>, axis:usize) -> ArrayD<f64> {

        let n:isize = x.len().try_into().unwrap(); // slice wants isize for some reason
        let dx = ArrayRef::diff(&x,1,Axis(0));

        let mut grad = field.to_owned(); // gradient result

        let f_p = &field.slice_axis(Axis(axis),Slice::new(2,Some(n),1)); // f_{i+1}
        let f_m = &field.slice_axis(Axis(axis),Slice::new(0,Some(n-2),1)); // f_{i-1}
        let f = &field.slice_axis(Axis(axis),Slice::new(1,Some(n-1),1)); // f_i
        
        let mut _d = dx.slice_axis(Axis(0),Slice::new(1,Some(n-1),1)).into_dyn();
        let mut _d_m = dx.slice_axis(Axis(0),Slice::new(0,Some(n-2),1)).into_dyn();

        for i in 0..(field.ndim()-1-axis) {
            _d = _d.insert_axis(Axis(0));
            _d_m = _d_m.insert_axis(Axis(0));
            println!("Added axis");
        }
        _d = _d.reversed_axes();
        _d_m = _d_m.reversed_axes();

        let d = &_d.broadcast(f.shape()).unwrap();
        let d_m = &_d_m.broadcast(f.shape()).unwrap();


        grad.slice_axis_mut(Axis(axis),Slice::new(1,Some(n-1),1)) // oder 2 inner region
        .assign(&((d_m*d_m *f_p - (d_m *d_m - d*d)*f -d*d*f_m)/(d*d_m*(d+d_m))));

        //Order 1 forward and backward for edges
        let f_0 = &field.slice_axis(Axis(axis),Slice::new(0,Some(1),1));
        let f_1 = &field.slice_axis(Axis(axis),Slice::new(1,Some(2),1));

        grad.slice_axis_mut(Axis(axis),Slice::new(0,Some(1),1)) 
        .assign(&((f_1-f_0)/dx[[0]]));

        let f_m1 = &field.slice_axis(Axis(axis),Slice::new(n-1,Some(n),1));
        let f_m2 = &field.slice_axis(Axis(axis),Slice::new(n-2,Some(n-1),1));

        grad.slice_axis_mut(Axis(axis),Slice::new(n-1,Some(n),1)) 
        .assign(&((f_m1-f_m2)/dx[[x.len()-2]]));


        grad
    }

    #[pyfunction(name="gradient")]
    fn gradient_py<'py>( py: Python<'py>,
                        field: &Bound<'py, PyArrayDyn<f64>>,
                        x: &Bound<'py, PyArray1<f64>>,
                        axis:Bound<'py, PyInt>
                    ) -> Bound<'py, PyArrayDyn<f64>> {
        let ff = unsafe {field.as_array()};
        let xx = unsafe {x.as_array()};
        let a = axis.extract::<usize>();
        gradient(&ff,&xx,a.expect("LJFLs")).into_pyarray(py) // .expect(...) solution given by the compiler but why ???
    }

    fn compute_velocity_divergence_shearing_box(vx1: &ArrayViewD<'_, f64>,
                                vx2: &ArrayViewD<'_, f64>,
                                vx3: &ArrayViewD<'_, f64>,
                                x:&ArrayView1<'_, f64>,
                                y:&ArrayView1<'_, f64>,
                                z:&ArrayView1<'_, f64>,
                                q: f64,
                                omega :f64,) -> ArrayD< f64> {
    
    let (xx,_yy,_zz) = meshgrid((x,y,z),MeshIndex::IJ); //requieres unreleased nupy-rust to acces ndarray >=0.17

    
    &gradient(vx1,x,0)+ &gradient(&(vx2 + q * omega * &xx).view(),y,1)+ &gradient(vx3 ,z,2)

    }



    fn find_adi_shearing_box(rho: &ArrayViewD<'_, f64>,
            vx1: &ArrayViewD<'_, f64>,
            vx2: &ArrayViewD<'_, f64>,
            vx3: &ArrayViewD<'_, f64>,
            prs: &ArrayViewD<'_, f64>,
            phi: &ArrayViewD<'_, f64>,
            q: f64,
            omega :f64,
            gamma: f64,
            x:&ArrayView1<'_, f64>,
            y:&ArrayView1<'_, f64>,
            z:&ArrayView1<'_, f64>,
            e_thresh: f64,
            rho_thresh: f64) -> Vec<usize>{
                /// this returns the list of indexes (truples) where the criterion is satisfied
    
                let e_tot = compute_total_energy_adi_shearing_box(rho,vx1,vx2,vx3,prs,phi,q,omega,gamma,x,y,z);
                let div = compute_velocity_divergence_shearing_box(vx1,vx2,vx3,x,y,z,q,omega);

                let mask = (&e_tot).map(|x| *x < e_thresh) & (&div).map(|x| *x < 0.0) & (rho).map(|x| *x > rho_thresh);
                
                mask.iter()
                .enumerate()
                .filter_map(|(index, &value)| (value == true).then(|| index))
                .collect() // ca renvoie peut-etre des indices 1D, à vérifier
        
    }

}
