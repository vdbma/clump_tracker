use pyo3::prelude::*;

/// A Python module implemented in Rust. The name of this module must match
/// the `lib.name` setting in the `Cargo.toml`, else Python will not be able to
/// import the module.
#[pymodule]
mod _core {
    use std::cmp::min;
    use std::string::String;
    use pyo3::prelude::*;
    use pyo3::types::{PyDict,PyInt,PyList,PyFloat,PyString};
    use numpy::ndarray::{Array,ArrayRef, Array1, Array2,ArrayD, ArrayView1,ArrayView3, ArrayViewD, ArrayViewMutD,meshgrid, MeshIndex, Axis, Zip,Slice,IxDyn};
    use numpy::{IntoPyArray, PyArray1, PyArray2, PyArrayDyn, PyReadonlyArrayDyn, PyArrayMethods};
    use num_traits::{Float, Signed};
    use std::ops::{Add, Sub, Mul, Div, AddAssign, SubAssign, MulAssign, DivAssign,Neg};

    trait AtLeastF32: Float + From<f32> + Signed +numpy::ndarray::ScalarOperand
    + Copy
    + Clone
    + Add<Output = Self>
    + Sub<Output = Self>
    + Mul<Output = Self>
    + Div<Output = Self>
    + AddAssign
    + SubAssign
    + MulAssign
    + DivAssign
    + Neg<Output = Self>
{
}
    impl AtLeastF32 for f32 {}
    impl AtLeastF32 for f64 {}



    fn gradient<T:AtLeastF32>(field: &ArrayViewD<'_, T>, x: &ArrayView1<'_, T>, axis:usize) -> ArrayD<T> {

        let n:isize = x.len().try_into().unwrap(); // slice wants isize for some reason
        let mut grad = ArrayD::<T>::zeros(field.raw_dim());

        let dx = ArrayRef::diff(&x,1,Axis(0)); //why mt though ?


        let f_p = &field.slice_axis(Axis(axis),Slice::new(2,Some(n),1)); // f_{i+1}
        let f_m = &field.slice_axis(Axis(axis),Slice::new(0,Some(n-2),1)); // f_{i-1}
        let f = &field.slice_axis(Axis(axis),Slice::new(1,Some(n-1),1)); // f_i

        let mut dims = Array1::ones::<usize>(field.ndim()-axis);
        dims[[0]] = n-2;

        let _dims = dims.mapv(|x| x as usize);
        let shape = IxDyn(_dims.as_slice().unwrap());

        let mut _d = ArrayD::zeros(shape.clone() );
        let mut _d_m = ArrayD::zeros(shape.clone() );

        _d.slice_axis_mut(Axis(0),Slice::new(0,Some(n-2),1))
        .assign(&dx.slice_axis(Axis(0),Slice::new(1,Some(n-1),1)).to_shape(shape.clone()).unwrap());

        _d_m.slice_axis_mut(Axis(0),Slice::new(0,Some(n-2),1))
        .assign(&dx.slice_axis(Axis(0),Slice::new(0,Some(n-2),1)).to_shape(shape.clone()).unwrap());

        let d = &_d.broadcast(f.shape()).unwrap();
        let d_m = &_d_m.broadcast(f.shape()).unwrap();



        grad.slice_axis_mut(Axis(axis),Slice::new(1,Some(n-1),1)) // order 2 inner region
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

    #[pyfunction(name="gradient_f32")]
    fn gradient_f32_py<'py>( py: Python<'py>,
                        field: &Bound<'py, PyArrayDyn<f32>>,
                        x: &Bound<'py, PyArray1<f32>>,
                        axis:Bound<'py, PyInt>
                    ) -> Bound<'py, PyArrayDyn<f32>> {
        let ff = unsafe {field.as_array()};
        let xx = unsafe {x.as_array()};
        let a = axis.extract::<usize>();
        gradient::<f32>(&ff,&xx,a.expect("LJFLs")).into_pyarray(py) // .expect(...) solution given by the compiler but why ???
    }

    #[pyfunction(name="gradient_f64")]
    fn gradient_f64_py<'py>( py: Python<'py>,
                        field: &Bound<'py, PyArrayDyn<f64>>,
                        x: &Bound<'py, PyArray1<f64>>,
                        axis:Bound<'py, PyInt>
                    ) -> Bound<'py, PyArrayDyn<f64>> {
        let ff = unsafe {field.as_array()};
        let xx = unsafe {x.as_array()};
        let a = axis.extract::<usize>();
        gradient::<f64>(&ff,&xx,a.expect("LJFLs")).into_pyarray(py) // .expect(...) solution given by the compiler but why ???
    }

    /*
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
                        // DO NOT USE, not ready
    let (xx,_yy,_zz) = meshgrid((x,y,z),MeshIndex::IJ); //requires unreleased nupy-rust to access ndarray >=0.17

    let e_c = 0.5 * rho * (vx1 * vx1 + (vx2 + q * omega * &xx) * (vx2 + q * omega * &xx) + vx3 * vx3);
    let e_th = prs / (gamma -1.);
    let e_p = rho * phi;
    &e_th + &e_c + &e_p
    }

    fn compute_velocity_divergence_shearing_box(vx1: &ArrayViewD<'_, f64>,
                                vx2: &ArrayViewD<'_, f64>,
                                vx3: &ArrayViewD<'_, f64>,
                                x:&ArrayView1<'_, f64>,
                                y:&ArrayView1<'_, f64>,
                                z:&ArrayView1<'_, f64>,
                                q: f64,
                                omega :f64,) -> ArrayD< f64> {

    let (xx,_yy,_zz) = meshgrid((x,y,z),MeshIndex::IJ); //requires unreleased nupy-rust to access ndarray >=0.17


    &gradient(vx1,x,0)+ &gradient(&(vx2 + q * omega * &xx).view(),y,1)//+ &gradient(vx3 ,z,2)

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

                let mask = (&e_tot).map(|x| *x < e_thresh) & (&div).map(|x| *x < 0.0);// & (rho).map(|x| *x > rho_thresh);

                mask.iter()
                .enumerate()
                .filter_map(|(index, &value)| (value == true).then(|| index))
                .collect() // ca renvoie peut-etre des indices 1D, à vérifier

    }

    #[pyfunction(name="find_adi_shearing_box")]
    fn find_adi_shearing_box_py<'py>( py: Python<'py>,
                        rho: &Bound<'py, PyArrayDyn<f64>>,
                        vx1: &Bound<'py, PyArrayDyn<f64>>,
                        vx2: &Bound<'py, PyArrayDyn<f64>>,
                        vx3: &Bound<'py, PyArrayDyn<f64>>,
                        prs: &Bound<'py, PyArrayDyn<f64>>,
                        phi: &Bound<'py, PyArrayDyn<f64>>,
                        q: Bound<'_, PyFloat>,
                        omega:Bound<'_, PyFloat>,
                        gamma:Bound<'_, PyFloat>,
                        x: &Bound<'py, PyArray1<f64>>,
                        y: &Bound<'py, PyArray1<f64>>,
                        z: &Bound<'py, PyArray1<f64>>,
                        e_thresh: Bound<'_, PyFloat>,
                        rho_thresh: Bound<'_, PyFloat>
                    ) ->Bound<'py, PyList> {

        let a_rho = unsafe {rho.as_array()};
        let a_vx1 = unsafe {vx1.as_array()};
        let a_vx2 = unsafe {vx2.as_array()};
        let a_vx3 = unsafe {vx3.as_array()};
        let a_prs = unsafe {prs.as_array()};
        let a_phi = unsafe {phi.as_array()};
        let a_x = unsafe {x.as_array()};
        let a_y = unsafe {y.as_array()};
        let a_z = unsafe {z.as_array()};

        let f_q:f64 = q.extract().unwrap();
        let f_omega:f64 = omega.extract().unwrap();
        let f_gamma:f64 = gamma.extract().unwrap();
        let f_e_thresh:f64 = e_thresh.extract().unwrap();
        let f_rho_thresh:f64 = rho_thresh.extract().unwrap();

        PyList::new(py,find_adi_shearing_box(&a_rho,&a_vx1,&a_vx2,&a_vx3,&a_prs,&a_phi,f_q,f_omega,f_gamma,&a_x,&a_y,&a_z,f_e_thresh,f_rho_thresh))
        .expect("LJFLs")
    }
    */

    fn compute_adjacency_cartesian<T:AtLeastF32>(indexes:&Vec<Vec<usize>>,
        x:&ArrayView1<'_, T>,
        y:&ArrayView1<'_, T>,
        z:&ArrayView1<'_, T>,
        d: T ) -> Array2<u8> {
        // indexes : list of 3D indexes
        // d : distance to be considered connected

        let mut adj = Array2::<u8>::zeros((indexes.len(),indexes.len())); // no bool arra :()

        for i in 0..indexes.len() {
            for j in 0..indexes.len() {
                let d2 = (x[indexes[i][0]]-x[indexes[j][0]])*(x[indexes[i][0]]-x[indexes[j][0]])
                        +(y[indexes[i][1]]-y[indexes[j][1]])*(y[indexes[i][1]]-y[indexes[j][1]])
                        +(z[indexes[i][2]]-z[indexes[j][2]])*(z[indexes[i][2]]-z[indexes[j][2]])  ;
                adj[[i,j]] = (d2 <= d*d).try_into().unwrap();
            }
        }
        adj
    }

    #[pyfunction(name="compute_adjacency_cartesian_f32")]
    fn compute_adjacency_cartesian_f32_py<'py>( py: Python<'py>,
                            indexes:Bound<'_, PyList>,
                            x: &Bound<'py, PyArray1<f32>>,
                            y: &Bound<'py, PyArray1<f32>>,
                            z: &Bound<'py, PyArray1<f32>>,
                            d: Bound<'_, PyFloat>
                        ) -> Bound<'py, PyArray2<u8>> {
        let r_x = unsafe {x.as_array()};
        let r_y = unsafe {y.as_array()};
        let r_z = unsafe {z.as_array()};
        let r_d:f32 = d.extract().unwrap();
        let r_indexes:Vec<Vec<usize>> = indexes.extract().unwrap();

        compute_adjacency_cartesian::<f32>(&r_indexes,&r_x,&r_y,&r_z,r_d).into_pyarray(py)
    }

    #[pyfunction(name="compute_adjacency_cartesian_f64")]
    fn compute_adjacency_cartesian_f64_py<'py>( py: Python<'py>,
                            indexes:Bound<'_, PyList>,
                            x: &Bound<'py, PyArray1<f64>>,
                            y: &Bound<'py, PyArray1<f64>>,
                            z: &Bound<'py, PyArray1<f64>>,
                            d: Bound<'_, PyFloat>
                        ) -> Bound<'py, PyArray2<u8>> {
        let r_x = unsafe {x.as_array()};
        let r_y = unsafe {y.as_array()};
        let r_z = unsafe {z.as_array()};
        let r_d:f64 = d.extract().unwrap();
        let r_indexes:Vec<Vec<usize>> = indexes.extract().unwrap();

        compute_adjacency_cartesian::<f64>(&r_indexes,&r_x,&r_y,&r_z,r_d).into_pyarray(py)
    }

    fn compute_cc<T:AtLeastF32>(indexes:&Vec<Vec<usize>>,
        x:&ArrayView1<'_, T>,
        y:&ArrayView1<'_, T>,
        z:&ArrayView1<'_, T>,
        d: T,
        geometry:String ) -> Vec<Vec<usize>> {


    let mut composante_connexes:Vec<Vec<usize>> = Vec::new();
    let mut nb_cc:usize = 0;

    let mut deja_vus = Array1::<u8>::zeros(indexes.len());

    let adj;
    if geometry == "cartesian" {
            adj = compute_adjacency_cartesian::<T>(indexes,x,y,z,d);
    } else {
        unimplemented!()
    }


    for (i,_) in indexes.iter().enumerate(){
        let mut a_visiter = Array1::<u8>::zeros(indexes.len());
        for k in 0..indexes.len() {
            a_visiter[k] = adj[[i,k]];
        }
        a_visiter[[i]] = 0;
        deja_vus[[i]] = 1;

        for k in 0..indexes.len() {
            if a_visiter[k] == 1 && deja_vus[k] == 0 {
                a_visiter[k] = 1;
            } else {
                a_visiter[k] = 0; // useless
            }
        }

        composante_connexes.push([i].to_vec());
        nb_cc +=1;

        while a_visiter.sum()>0 {
            for (j,c) in indexes.iter().enumerate() {
                if deja_vus[j] == 0 {
                    for k in 0..indexes.len() {
                        a_visiter[k] += adj[[i,k]];
                        a_visiter[k] = min(a_visiter[k],1);
                    }
                    a_visiter[j] = 0;
                    deja_vus[j] = 1;

                    for k in 0..indexes.len() {
                        if a_visiter[k] == 1 && deja_vus[k] == 0 {
                            a_visiter[k] = 1;
                        } else {
                            a_visiter[k] = 0;
                        }
                    }
                    composante_connexes[nb_cc-1].push(j);
                }
            }
        }

        let nb_deja_vus:usize = deja_vus.sum().try_into().unwrap();
        if  nb_deja_vus== indexes.len(){
            break;
        }


    }

    composante_connexes
    }

    #[pyfunction(name="compute_cc_f32")]
    fn compute_cc_f32_py<'py>( py: Python<'py>,
                            indexes:Bound<'_, PyList>,
                            x: &Bound<'py, PyArray1<f32>>,
                            y: &Bound<'py, PyArray1<f32>>,
                            z: &Bound<'py, PyArray1<f32>>,
                            d: Bound<'_, PyFloat>,
                            geometry: Bound<'_, PyString>
                        ) -> Bound<'py, PyList> {
        let r_x = unsafe {x.as_array()};
        let r_y = unsafe {y.as_array()};
        let r_z = unsafe {z.as_array()};
        let r_d:f32 = d.extract().unwrap();
        let r_indexes:Vec<Vec<usize>> = indexes.extract().unwrap();
        let r_geometry:String = geometry.extract().unwrap();

        PyList::new(py,compute_cc::<f32>(&r_indexes,&r_x,&r_y,&r_z,r_d,r_geometry)).expect("???")
    }

    #[pyfunction(name="compute_cc_f64")]
    fn compute_cc_f64_py<'py>( py: Python<'py>,
                            indexes:Bound<'_, PyList>,
                            x: &Bound<'py, PyArray1<f64>>,
                            y: &Bound<'py, PyArray1<f64>>,
                            z: &Bound<'py, PyArray1<f64>>,
                            d: Bound<'_, PyFloat>,
                            geometry: Bound<'_, PyString>
                        ) -> Bound<'py, PyList> {
        let r_x = unsafe {x.as_array()};
        let r_y = unsafe {y.as_array()};
        let r_z = unsafe {z.as_array()};
        let r_d:f64 = d.extract().unwrap();
        let r_indexes:Vec<Vec<usize>> = indexes.extract().unwrap();
        let r_geometry:String = geometry.extract().unwrap();

        PyList::new(py,compute_cc::<f64>(&r_indexes,&r_x,&r_y,&r_z,r_d,r_geometry)).expect("???")
    }
}
